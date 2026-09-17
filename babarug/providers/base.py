"""Interfaces de providers.

Objectif : pouvoir changer de modele ou de fournisseur sans toucher au pipeline.
Chaque etape ne connait qu'un protocole, jamais un SDK. Les implementations
concretes vivent dans les fichiers voisins et sont selectionnees par nom dans
`registry.py`.

Les quatre roles du cahier des charges :
  VisionProvider          -> lecture des photos, RUG DNA, lecture de la scene
  ImageGenerationProvider -> fabrication du decor VIDE
  ImageEditingProvider    -> retouche locale optionnelle (jamais sur le tapis)
  QualityControlProvider  -> avis de scene, consultatif
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from babarug.models import RugDNA, SceneBrief, SceneReview


@dataclass
class ProviderCost:
    """Comptabilite par appel. Permet d'afficher un cout reel par tapis."""

    provider: str
    operation: str
    usd: float
    detail: str = ""


class ProviderError(RuntimeError):
    """Echec d'un provider. Le pipeline decide s'il retente ou degrade."""


@dataclass
class FloorEstimate:
    """Lecture du plan du sol dans une scene generee."""

    quad_norm: tuple  # 4 points (x, y) normalises, ordre TL, TR, BR, BL
    width_m: float
    depth_m: float
    confidence: float
    notes: str = ""


@runtime_checkable
class VisionProvider(Protocol):
    def analyze_rug(
        self, product_id: str, images: list[tuple[str, bytes]], declared_size_cm=None
    ) -> RugDNA:
        """Les N photos -> UN RugDNA. Un seul appel : le modele doit voir toutes
        les vues ensemble pour distinguer le produit de la prise de vue."""
        ...

    def locate_floor(self, scene_png: bytes, brief: SceneBrief) -> FloorEstimate:
        """Ou poser le tapis, et quelle est l'echelle reelle de cette zone."""
        ...

    @property
    def costs(self) -> list[ProviderCost]: ...


@runtime_checkable
class ImageGenerationProvider(Protocol):
    def generate_scene(self, brief: SceneBrief, width: int, height: int, seed: int) -> bytes:
        """Genere la piece VIDE. Le tapis n'est jamais demande au generateur."""
        ...

    @property
    def costs(self) -> list[ProviderCost]: ...


@runtime_checkable
class ImageEditingProvider(Protocol):
    def edit(self, image_png: bytes, mask_png: bytes, prompt: str, refs=None) -> bytes:
        """Retouche masquee. Le masque doit EXCLURE le tapis : cette etape n'a
        pas le droit de toucher au produit."""
        ...

    @property
    def costs(self) -> list[ProviderCost]: ...


@runtime_checkable
class SegmentationProvider(Protocol):
    def segment_rug(self, image_bgr: np.ndarray) -> np.ndarray:
        """Masque du tapis, franges comprises. uint8 0/255."""
        ...

    @property
    def costs(self) -> list[ProviderCost]: ...


@runtime_checkable
class QualityControlProvider(Protocol):
    def review_scene(self, final_png: bytes, brief: SceneBrief, dna: RugDNA) -> SceneReview:
        """Avis sur la SCENE. Consultatif : ne peut jamais valider un echec de
        fidelite mesure."""
        ...

    @property
    def costs(self) -> list[ProviderCost]: ...


class CostTracker:
    """Melange-moi dans un provider pour obtenir `.costs` gratuitement."""

    def __init__(self):
        self._costs: list[ProviderCost] = []

    def _charge(self, operation: str, usd: float, detail: str = ""):
        self._costs.append(ProviderCost(type(self).__name__, operation, usd, detail))

    @property
    def costs(self) -> list[ProviderCost]:
        return list(self._costs)

    @property
    def total_usd(self) -> float:
        return sum(c.usd for c in self._costs)
