"""Contrats de donnees du pipeline Baba Rug.

Le RUG DNA est la piece centrale : il est produit une seule fois par tapis, a
partir des N photos, et sert ensuite de reference a toutes les etapes (brief de
scene, echelle du compositing, controle qualite). Tout ce qui est "identite du
tapis" vit ici ; tout ce qui est "gout / decor" vit dans SceneBrief.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

Quad = tuple[
    tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]
]
"""Quadrilatere TL, TR, BR, BL. Coordonnees normalisees [0,1] sur (largeur, hauteur)."""


class Shape(str, Enum):
    RECTANGULAR = "rectangular"
    RUNNER = "runner"
    SQUARE = "square"
    ROUND = "round"
    OVAL = "oval"
    IRREGULAR = "irregular"


class ViewKind(str, Enum):
    """Role photographique d'une image dans la serie."""

    FULL_FLAT = "full_flat"      # tapis entier, vu de dessus / faible angle -> candidat plate
    FULL_ANGLE = "full_angle"    # tapis entier, en perspective marquee
    DETAIL = "detail"            # motif, noeud, texture
    BORDER = "border"            # bordure
    FRINGE = "fringe"            # franges
    BACK = "back"                # dos / envers
    EDGE = "edge"                # epaisseur, tranche
    OTHER = "other"


class ColorRef(BaseModel):
    name: str = Field(description="Nom courant, ex: 'rouge brique'")
    hex: str = Field(description="Couleur echantillonnee, ex: '#8C3B2E'")
    coverage: float = Field(ge=0, le=1, description="Part approximative de la surface")


class PhotoAnalysis(BaseModel):
    """Lecture d'UNE photo. Sert a choisir la plate et a separer produit / prise de vue."""

    filename: str
    view: ViewKind
    shows_full_rug: bool
    rug_coverage: float = Field(ge=0, le=1, description="Part de l'image occupee par le tapis")
    perspective_severity: float = Field(
        ge=0, le=1, description="0 = vue de dessus, 1 = rasant. Bas = meilleure plate"
    )
    sharpness: float = Field(ge=0, le=1)
    color_cast: str = Field(description="Dominante lumiere, ex: 'neutre', 'chaud/tungstene'")
    usable_as_plate: bool = Field(
        description="Le tapis entier est visible, a plat, peu deforme, non occlus"
    )
    notes: str = ""


class RugDNA(BaseModel):
    """Identite visuelle consolidee du tapis, commune aux N photos."""

    product_id: str
    n_photos: int

    # --- geometrie ---
    shape: Shape
    aspect_ratio: float = Field(gt=0, description="longueur / largeur")
    aspect_ratio_confidence: float = Field(ge=0, le=1)
    orientation: Literal["portrait", "landscape", "square"]
    real_size_cm: tuple[float, float] | None = Field(
        default=None, description="(longueur, largeur) si connue/declaree. Pilote l'echelle."
    )

    # --- couleur ---
    dominant_colors: list[ColorRef]
    secondary_colors: list[ColorRef]

    # --- dessin ---
    pattern_description: str
    border_description: str
    central_motif: str
    repetition: str = Field(description="Rythme des repetitions, symetries, sens de lecture")

    # --- matiere ---
    fringe: str
    texture: str
    material_guess: str
    pile_height: str = ""

    # --- ce qui rend CE tapis unique ---
    wear: str
    irregularities: list[str] = Field(
        default_factory=list,
        description="Asymetries, abrash, bords non droits, reparations, taches. Ne pas corriger.",
    )
    unique_features: list[str] = Field(default_factory=list)

    # --- tracabilite ---
    photos: list[PhotoAnalysis]
    plate_photo: str = Field(description="Photo retenue comme plate (orthophoto source)")
    photographic_artifacts: list[str] = Field(
        default_factory=list,
        description="Variations dues a la lumiere/optique, PAS au produit. A ignorer en QC.",
    )
    analysis_notes: str = ""

    @property
    def needs_human_check(self) -> bool:
        """Vrai si le DNA est trop incertain pour lancer une generation en aveugle."""
        return (
            self.aspect_ratio_confidence < 0.6
            or not any(p.usable_as_plate for p in self.photos)
            or self.shape == Shape.IRREGULAR
        )


class SceneBrief(BaseModel):
    """Le decor demande. Ne contient JAMAIS de description du tapis :
    le tapis n'est pas genere, il est compose."""

    slug: str
    interior_style: str
    room_type: str
    style_key: str = ""   # cle du catalogue, ex "parisien_contemporain"
    room_key: str = ""    # cle du catalogue, ex "salon"
    flooring: str
    key_furniture: list[str]
    lighting: str
    camera_height_m: float
    camera_pitch_deg: float = Field(description="Negatif = plongee")
    focal_mm: int
    mood: str
    negative: str = ""

    def prompt(self) -> str:
        """Prompt 'piece vide'. Le sol doit rester degage a l'emplacement du tapis."""
        furn = ", ".join(self.key_furniture)
        return (
            f"Interior photograph of a {self.interior_style} {self.room_type}. "
            f"{self.flooring}. Furniture: {furn}. {self.lighting}. {self.mood}. "
            f"Shot on a {self.focal_mm}mm lens, camera {self.camera_height_m:.2f}m high, "
            f"{abs(self.camera_pitch_deg):.0f} degrees "
            f"{'downward' if self.camera_pitch_deg < 0 else 'upward'} tilt. "
            "IMPORTANT: the floor area in front of the main seating is COMPLETELY BARE — "
            "no rug, no carpet, no mat, no floor covering of any kind. "
            "Nothing lies on that floor area and nothing overlaps it. "
            "Editorial interior photography, natural materials, realistic architecture, "
            "photorealistic, no CGI look."
        )


class Verdict(str, Enum):
    APPROVED = "APPROVED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    REJECTED = "REJECTED"


class FidelityMetrics(BaseModel):
    """Mesures DETERMINISTES, calculees en re-extrayant le tapis de l'image finale
    et en le recomparant a la plate source. Aucun modele n'intervient ici."""

    delta_e_mean: float = Field(description="CIEDE2000 moyen. < 2 = imperceptible")
    delta_e_p95: float
    ssim: float = Field(description="Similarite structurelle luminance, 0..1")
    keypoint_inlier_ratio: float = Field(description="Appariement ORB plate <-> rendu, 0..1")
    histogram_correlation: float
    visible_fraction: float = Field(description="Part du tapis non occultee dans le rendu")
    scale_error: float = Field(description="Ecart relatif du ratio L/l restitue")
    texture_retention: float = Field(
        default=1.0,
        description="Energie haute frequence conservee. <1 = texture lissee, "
        "usure effacee, tapis rendu artificiellement neuf",
    )
    silhouette_iou: float = Field(
        default=1.0,
        description="IoU entre la silhouette attendue et celle rendue. Chute si "
        "les franges ou les bords irreguliers ont ete rabotes",
    )

    @property
    def summary(self) -> str:
        return (
            f"dE={self.delta_e_mean:.2f}/{self.delta_e_p95:.2f} ssim={self.ssim:.3f} "
            f"kp={self.keypoint_inlier_ratio:.2f} vis={self.visible_fraction:.2f} "
            f"tex={self.texture_retention:.2f} sil={self.silhouette_iou:.3f}"
        )


class SceneReview(BaseModel):
    """Avis du modele sur la SCENE uniquement. Consultatif, jamais bloquant a la hausse :
    il ne peut pas transformer un echec de fidelite mesure en succes."""

    scene_realism: int = Field(ge=0, le=100)
    perspective: int = Field(ge=0, le=100)
    composition: int = Field(ge=0, le=100)
    integration: int = Field(ge=0, le=100, description="Le tapis a-t-il l'air pose, ou colle ?")
    rug_fully_visible: bool
    rug_merges_with_furniture: bool
    problems: list[str] = Field(default_factory=list)
    suggested_fix: str = ""


class QCReport(BaseModel):
    variant: str
    verdict: Verdict
    fidelity: FidelityMetrics
    review: SceneReview | None = None
    failures: list[str] = Field(default_factory=list)
    iteration: int = 0


class RenderResult(BaseModel):
    variant: str
    image_path: str
    export_path: str | None = None
    brief: SceneBrief
    qc: QCReport
    iterations: int
    cost_usd: float = 0.0
    seconds: float = 0.0
