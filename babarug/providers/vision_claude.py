"""VisionProvider + QualityControlProvider bases sur Claude.

Point cle : les N photos partent dans UN SEUL appel. C'est ce qui permet au
modele de repondre a la question posee dans le cahier des charges -- "quelles
caracteristiques sont communes aux N photos et appartiennent reellement a ce
tapis ?" -- au lieu de produire N descriptions independantes qu'il faudrait
ensuite reconcilier a l'aveugle.
"""

from __future__ import annotations

import base64
import json
import logging

import anthropic

from babarug.models import RugDNA, SceneBrief, SceneReview
from babarug.providers.base import CostTracker, FloorEstimate, ProviderError

log = logging.getLogger(__name__)

MODEL = "claude-opus-5"

# Tarif au 2026-06 ($/Mtok). Sert au suivi de cout, pas a la facturation.
PRICE_IN, PRICE_OUT = 5.0, 25.0

DNA_SYSTEM = """Tu es l'expert produit de Baba Rug, maison specialisee en tapis d'Orient, \
kilims et tapis noues main.

On te donne PLUSIEURS photos D'UN SEUL ET MEME TAPIS : vue principale, angles, \
details de motif, texture, bordure, franges, parfois le dos.

Ta tache : produire la fiche d'identite visuelle de CE tapis precis.

Regles imperatives :
1. Les photos montrent UN seul tapis. Ne decris jamais plusieurs tapis.
2. Separe rigoureusement :
   - ce qui appartient au PRODUIT (motif, couleurs de la laine, bordure, usure, \
asymetries, reparations, franges, matiere) ;
   - ce qui appartient a la PRISE DE VUE (dominante de lumiere, ombres portees, \
flou, angle, reflets, sous/sur-exposition, fond).
   Les secondes vont dans `photographic_artifacts` et JAMAIS dans la description produit.
3. Les defauts sont des donnees, pas des problemes. Usure, abrash, bords non droits, \
franges inegales, reparations, taches : decris-les precisement. Ils identifient ce tapis.
   N'ecris jamais qu'un tapis est "parfait", "impeccable" ou "comme neuf" s'il ne l'est pas.
4. Les couleurs doivent etre celles de la LAINE, pas celles de l'eclairage. Si une photo \
est jaune et une autre neutre, la verite est proche de la neutre : dis-le.
5. `aspect_ratio` = longueur / largeur, toujours >= 1 pour un format allonge. Si les \
photos ne permettent pas de trancher, baisse `aspect_ratio_confidence`. Ne devine pas \
avec assurance : une dimension fausse deforme le tapis dans toutes les images finales.
6. `usable_as_plate` = vrai uniquement si le tapis est ENTIER, a plat, peu deforme par \
la perspective, net, non occulte, non plie. C'est cette photo qui fournira les pixels \
reels du rendu final : sois severe.

Reponds uniquement par le JSON demande."""

REVIEW_SYSTEM = """Tu evalues une photographie d'interieur produite pour un site \
e-commerce de tapis d'Orient haut de gamme.

Le tapis present dans l'image est le VRAI produit, incruste geometriquement. Tu n'as pas \
a juger sa fidelite : elle est mesuree ailleurs, par des metriques deterministes.

Juge uniquement LA SCENE et L'INTEGRATION :
- realisme de la piece (architecture, materiaux, mobilier credibles)
- coherence de la perspective entre le tapis et le sol
- composition editoriale
- integration : le tapis a-t-il l'air POSE au sol, ou COLLE par-dessus ? Regarde l'ombre \
de contact, la coherence de l'eclairage, la nettete relative
- le tapis est-il entierement visible, ou coupe / fusionne avec un meuble ?

Penalise : rendu CGI evident, interieur generique d'IA, mobilier futuriste, luxe artificiel, \
piece surchargee, plastique, textures synthetiques.
Valorise : lumiere naturelle, materiaux naturels, architecture credible, sobriete.

Sois exigeant et concret. Un probleme vague n'aide personne."""


def _b64(data: bytes) -> str:
    return base64.standard_b64encode(data).decode()


def _media_type(filename: str) -> str:
    f = filename.lower()
    if f.endswith(".png"):
        return "image/png"
    if f.endswith(".webp"):
        return "image/webp"
    return "image/jpeg"


class ClaudeVisionProvider(CostTracker):
    """Implemente VisionProvider et QualityControlProvider."""

    def __init__(self, client: anthropic.Anthropic | None = None, model: str = MODEL):
        super().__init__()
        self.client = client or anthropic.Anthropic()
        self.model = model

    # ------------------------------------------------------------------ couts
    def _track(self, op: str, usage):
        usd = (usage.input_tokens * PRICE_IN + usage.output_tokens * PRICE_OUT) / 1e6
        self._charge(op, usd, f"in={usage.input_tokens} out={usage.output_tokens}")

    def _parse(self, model_cls, response):
        try:
            text = next(b.text for b in response.content if b.type == "text")
        except StopIteration:
            raise ProviderError("reponse sans bloc texte")
        try:
            return model_cls.model_validate(json.loads(text))
        except Exception as e:
            raise ProviderError(f"JSON invalide pour {model_cls.__name__}: {e}") from e

    # ------------------------------------------------------------------ RUG DNA
    def analyze_rug(self, product_id, images, declared_size_cm=None) -> RugDNA:
        if not images:
            raise ProviderError("aucune photo fournie")

        content = []
        for name, data in images:
            content.append(
                {"type": "text", "text": f"--- photo : {name} ---"}
            )
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": _media_type(name),
                        "data": _b64(data),
                    },
                }
            )

        size_line = (
            f"Dimensions reelles declarees : {declared_size_cm[0]} x {declared_size_cm[1]} cm. "
            "Ces dimensions font foi : deduis-en `aspect_ratio` et mets "
            "`aspect_ratio_confidence` a 1.0."
            if declared_size_cm
            else "Dimensions reelles non fournies : estime le ratio depuis les photos et "
            "sois honnete sur ta confiance."
        )
        content.append(
            {
                "type": "text",
                "text": (
                    f"product_id = {product_id}\n"
                    f"Nombre de photos = {len(images)}\n"
                    f"{size_line}\n\n"
                    "Produis le RUG DNA. Un objet `PhotoAnalysis` par photo, dans l'ordre, "
                    "avec le nom de fichier exact donne ci-dessus."
                ),
            }
        )

        schema = RugDNA.model_json_schema()
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=16000,
                system=DNA_SYSTEM,
                thinking={"type": "adaptive"},
                output_config={"format": {"type": "json_schema", "schema": schema}},
                messages=[{"role": "user", "content": content}],
            )
        except anthropic.APIError as e:
            raise ProviderError(f"analyse du tapis echouee : {e}") from e

        self._track("analyze_rug", resp.usage)
        dna = self._parse(RugDNA, resp)

        # Les dimensions declarees priment toujours sur l'estimation du modele.
        if declared_size_cm:
            long_, short = max(declared_size_cm), min(declared_size_cm)
            dna.real_size_cm = (float(long_), float(short))
            dna.aspect_ratio = float(long_) / float(short)
            dna.aspect_ratio_confidence = 1.0
        return dna

    # ------------------------------------------------------------------ sol
    def locate_floor(self, scene_png: bytes, brief: SceneBrief) -> FloorEstimate:
        schema = {
            "type": "object",
            "properties": {
                "quad_norm": {
                    "type": "array",
                    "description": "4 points [x,y] normalises 0-1, ordre : "
                    "haut-gauche, haut-droit, bas-droit, bas-gauche",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
                "width_m": {"type": "number"},
                "depth_m": {"type": "number"},
                "confidence": {"type": "number"},
                "notes": {"type": "string"},
            },
            "required": ["quad_norm", "width_m", "depth_m", "confidence", "notes"],
            "additionalProperties": False,
        }
        prompt = (
            "Cette photo d'interieur a ete generee SANS tapis.\n\n"
            "1. Delimite la zone de SOL DEGAGEE ou un tapis devrait etre pose "
            f"(typiquement devant l'assise principale d'un {brief.room_type}).\n"
            "   Donne 4 points normalises, dans l'ordre haut-gauche, haut-droit, "
            "bas-droit, bas-gauche, en suivant la PERSPECTIVE du sol : le bord "
            "lointain est plus etroit que le bord proche.\n"
            "   La zone doit etre entierement du sol : aucun meuble, aucun mur.\n"
            "2. Estime ses dimensions reelles en metres (width_m = largeur gauche-droite, "
            "depth_m = profondeur). Sers-toi d'echelles connues : une assise de canape "
            "fait ~2 m de large, une lame de parquet ~12 cm, une porte ~2,05 m.\n"
            "3. confidence entre 0 et 1.\n\n"
            "L'echelle sert a poser un tapis a sa taille reelle : une erreur ici donne "
            "un tapis geant ou un paillasson."
        )
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                thinking={"type": "adaptive"},
                output_config={"format": {"type": "json_schema", "schema": schema}},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": _b64(scene_png),
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
        except anthropic.APIError as e:
            raise ProviderError(f"localisation du sol echouee : {e}") from e

        self._track("locate_floor", resp.usage)
        text = next(b.text for b in resp.content if b.type == "text")
        d = json.loads(text)
        quad = tuple((float(p[0]), float(p[1])) for p in d["quad_norm"][:4])
        if len(quad) != 4:
            raise ProviderError("le modele n'a pas renvoye 4 points de sol")
        return FloorEstimate(
            quad_norm=quad,
            width_m=float(d["width_m"]),
            depth_m=float(d["depth_m"]),
            confidence=float(d["confidence"]),
            notes=d.get("notes", ""),
        )

    # ------------------------------------------------------------------ avis scene
    def review_scene(self, final_png: bytes, brief: SceneBrief, dna: RugDNA) -> SceneReview:
        schema = SceneReview.model_json_schema()
        prompt = (
            f"Style demande : {brief.interior_style}, {brief.room_type}. "
            f"Sol : {brief.flooring}. Lumiere : {brief.lighting}.\n"
            f"Le tapis est un {dna.shape.value} de ratio {dna.aspect_ratio:.2f}.\n\n"
            "Evalue la scene et l'integration du tapis."
        )
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                system=REVIEW_SYSTEM,
                thinking={"type": "adaptive"},
                output_config={"format": {"type": "json_schema", "schema": schema}},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": _b64(final_png),
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
        except anthropic.APIError as e:
            raise ProviderError(f"revue de scene echouee : {e}") from e

        self._track("review_scene", resp.usage)
        return self._parse(SceneReview, resp)
