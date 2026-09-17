"""Catalogue de styles et moteur de variation des scenes.

Contrainte du cahier des charges : les deux images d'un meme tapis doivent etre
REELLEMENT differentes (architecture, mobilier, camera, lumiere) tout en gardant
le meme tapis. La variation est donc pilotee ici, de facon deterministe a partir
du product_id : deux lancements sur le meme tapis redonnent les memes scenes
(reproductible, debuggable), mais deux tapis differents ne se ressemblent pas.
"""

from __future__ import annotations

import hashlib
import random

from babarug.models import SceneBrief

INTERIOR_STYLES: dict[str, dict] = {
    "parisien_contemporain": {
        "label": "Parisien contemporain",
        "prompt": "contemporary Parisian apartment, elegant but restrained",
        "flooring": ["oak herringbone parquet", "point de Hongrie oak parquet"],
        "furniture": ["a linen sofa", "a low walnut coffee table", "a slim floor lamp",
                      "a pair of ceramic vases", "linen curtains"],
        "light": ["soft natural daylight from a tall window",
                  "late afternoon light raking across the floor"],
    },
    "haussmannien": {
        "label": "Haussmannien",
        "prompt": "Haussmannian apartment with moulded cornices and a marble fireplace",
        "flooring": ["point de Hongrie oak parquet", "aged oak parquet"],
        "furniture": ["a velvet sofa", "a marble fireplace", "gilt-framed mirror",
                      "a bergere armchair", "tall French windows"],
        "light": ["diffused daylight through tall French windows",
                  "soft overcast light"],
    },
    "mediterraneen": {
        "label": "Mediterraneen contemporain",
        "prompt": "contemporary Mediterranean interior, lime-plaster walls, arched openings",
        "flooring": ["pale terracotta tiles", "honed travertine floor", "wide pale oak boards"],
        "furniture": ["a boucle sofa", "an olive wood stool", "a rattan armchair",
                      "an earthenware jar", "a low travertine table"],
        "light": ["bright warm Mediterranean daylight", "dappled light through shutters"],
    },
    "maison_de_campagne": {
        "label": "Maison de campagne",
        "prompt": "French country house interior, honest materials, lived-in",
        "flooring": ["wide reclaimed oak boards", "aged terracotta tiles"],
        "furniture": ["a slipcovered sofa", "a solid oak farmhouse table",
                      "a woven basket", "a wooden bench"],
        "light": ["soft morning light through a small-paned window"],
    },
    "japandi": {
        "label": "Japandi",
        "prompt": "Japandi interior, calm, low furniture, warm minimal palette",
        "flooring": ["pale oak boards", "matte light timber floor"],
        "furniture": ["a low oak sofa", "a paper floor lamp", "a ceramic bowl",
                      "a low timber bench"],
        "light": ["even diffused daylight", "soft shoji-filtered light"],
    },
    "minimaliste": {
        "label": "Minimaliste",
        "prompt": "minimalist interior, few objects, generous empty surfaces",
        "flooring": ["polished concrete floor", "pale wide oak boards"],
        "furniture": ["a single low sofa", "one sculptural chair", "a plain plinth"],
        "light": ["cool north-facing daylight"],
    },
    "mid_century": {
        "label": "Mid-century",
        "prompt": "mid-century modern interior, teak and brass, 1950s-60s lines",
        "flooring": ["teak parquet", "warm oak boards"],
        "furniture": ["a teak-framed sofa", "an Eames-style lounge chair",
                      "a brass floor lamp", "a sideboard"],
        "light": ["warm afternoon sun through large panes"],
    },
    "vintage": {
        "label": "Vintage",
        "prompt": "warm vintage interior, collected furniture, patina",
        "flooring": ["worn oak parquet", "old terracotta tiles"],
        "furniture": ["a worn leather sofa", "a bookcase", "an antique wooden chest"],
        "light": ["warm lamplight mixed with window light"],
    },
    "industriel": {
        "label": "Brut / industriel",
        "prompt": "converted industrial loft, raw brick and steel-framed windows",
        "flooring": ["polished concrete floor", "dark stained wide boards"],
        "furniture": ["a deep linen sofa", "a steel and glass table", "a metal shelf"],
        "light": ["hard directional daylight through steel-framed windows"],
    },
    "boheme_chic": {
        "label": "Boheme chic",
        "prompt": "bohemian-chic interior, layered natural textiles, plants",
        "flooring": ["warm oak boards", "terracotta tiles"],
        "furniture": ["a low modular sofa", "floor cushions", "a large potted plant",
                      "a rattan pendant"],
        "light": ["warm filtered afternoon light"],
    },
}

ROOM_TYPES = {
    "salon": "living room",
    "salle_a_manger": "dining room",
    "chambre": "bedroom",
    "bureau": "home office",
    "entree": "entrance hall",
}

# Combinaisons volontairement ecartees : un tapis pose sous une table de salle a
# manger est masque par les pieds et les chaises, ce qui contredit "le tapis doit
# etre entierement visible". Mieux vaut interdire que corriger apres coup.
DISCOURAGED = {("minimaliste", "chambre"), ("industriel", "chambre"),
               ("haussmannien", "entree"), ("japandi", "entree")}

CAMERAS = [
    {"focal_mm": 35, "camera_height_m": 1.45, "camera_pitch_deg": -14, "mood": "wide editorial view"},
    {"focal_mm": 28, "camera_height_m": 1.60, "camera_pitch_deg": -22, "mood": "elevated three-quarter view"},
    {"focal_mm": 50, "camera_height_m": 1.25, "camera_pitch_deg": -8, "mood": "intimate eye-level view"},
    {"focal_mm": 24, "camera_height_m": 1.70, "camera_pitch_deg": -28, "mood": "high overview of the floor"},
    {"focal_mm": 40, "camera_height_m": 1.35, "camera_pitch_deg": -12, "mood": "balanced natural view"},
]

NEGATIVE = (
    "no CGI look, no 3D render, no generic AI interior, no futuristic furniture, "
    "no ostentatious luxury, no cluttered room, no plastic or synthetic textures, "
    "no distorted objects, no impossible proportions, no obvious artificial lighting"
)


def _rng(product_id: str, salt: str = "") -> random.Random:
    """Aleatoire reproductible : meme tapis -> memes scenes."""
    h = hashlib.sha256(f"{product_id}|{salt}".encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def build_brief(
    product_id: str, style_key: str, room_key: str, variant: str, seed_salt: str = ""
) -> SceneBrief:
    st = INTERIOR_STYLES[style_key]
    rng = _rng(product_id, f"{style_key}{room_key}{variant}{seed_salt}")
    cam = rng.choice(CAMERAS)
    n_furn = rng.randint(2, min(4, len(st["furniture"])))
    return SceneBrief(
        slug=f"{variant}_{style_key}_{room_key}",
        interior_style=st["prompt"],
        room_type=ROOM_TYPES[room_key],
        flooring=rng.choice(st["flooring"]),
        key_furniture=rng.sample(st["furniture"], n_furn),
        lighting=rng.choice(st["light"]),
        mood=cam["mood"],
        camera_height_m=cam["camera_height_m"],
        camera_pitch_deg=cam["camera_pitch_deg"],
        focal_mm=cam["focal_mm"],
        negative=NEGATIVE,
    )


def surprise_me(product_id: str, n: int = 2) -> list[tuple[str, str]]:
    """Choisit n combinaisons style/piece coherentes et FRANCHEMENT differentes.

    On impose que les deux scenes ne partagent ni le style ni la piece, et on
    ecarte les familles proches (parisien/haussmannien, japandi/minimaliste) :
    deux variations du meme salon beige ne repondent pas au besoin.
    """
    rng = _rng(product_id, "surprise")
    families = {
        "parisien_contemporain": "fr", "haussmannien": "fr",
        "japandi": "zen", "minimaliste": "zen",
        "mediterraneen": "sud", "boheme_chic": "sud",
        "maison_de_campagne": "rustique", "vintage": "rustique",
        "mid_century": "retro", "industriel": "brut",
    }
    picks: list[tuple[str, str]] = []
    used_fam: set[str] = set()
    used_room: set[str] = set()
    styles = list(INTERIOR_STYLES)
    rng.shuffle(styles)
    rooms = list(ROOM_TYPES)

    for s in styles:
        if len(picks) >= n:
            break
        if families[s] in used_fam:
            continue
        cand = [r for r in rooms if r not in used_room and (s, r) not in DISCOURAGED]
        if not cand:
            continue
        # Le salon montre le mieux un tapis : on le privilegie pour la 1re image.
        r = "salon" if ("salon" in cand and not picks) else rng.choice(cand)
        picks.append((s, r))
        used_fam.add(families[s])
        used_room.add(r)
    return picks
