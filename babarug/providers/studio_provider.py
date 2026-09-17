"""ImageGenerationProvider adosse au studio procedural.

Meme interface que les generateurs distants : le pipeline ne voit aucune
difference. Cout nul, aucune cle, resultat reproductible.

`default_floor` est fourni et EXACT : c'est nous qui posons la camera, donc la
zone degagee est projetee analytiquement au lieu d'etre devinee par un modele
de vision. C'est la raison principale pour laquelle ce mode est plus fiable
qu'une generation, en plus d'etre gratuit.
"""

from __future__ import annotations

import hashlib

import cv2
import numpy as np

from babarug.models import SceneBrief
from babarug.providers.base import CostTracker, FloorEstimate, ProviderError
from babarug.studio import STUDIO_STYLES, render_studio

# Correspondance entre le catalogue de styles d'interieur et les decors studio.
# Plusieurs styles peuvent partager un decor : ce qui compte est que les deux
# images d'un meme tapis ne tombent pas sur le meme.
STYLE_MAP: dict[str, str] = {
    "parisien_contemporain": "parisien_chene",
    "haussmannien": "parisien_chene",
    "mid_century": "atelier_chene_fonce",
    "vintage": "atelier_chene_fonce",
    "japandi": "lames_larges",
    "minimaliste": "lames_larges",
    "mediterraneen": "travertin_sud",
    "boheme_chic": "travertin_sud",
    "maison_de_campagne": "lames_larges",
    "industriel": "beton_brut",
}


class StudioProvider(CostTracker):
    """Decor fabrique par le code. Aucune API, aucun cout, plan du sol exact."""

    def __init__(self, product_id: str = "", **_):
        super().__init__()
        self.product_id = product_id
        self._used: set[str] = set()
        self._last: dict[str, tuple[str, FloorEstimate]] = {}

    def _pick(self, brief: SceneBrief) -> str:
        wanted = STYLE_MAP.get(brief.style_key, "parisien_chene")
        if wanted in self._used:
            libres = [k for k in STUDIO_STYLES if k not in self._used]
            if libres:
                h = hashlib.sha256(f"{self.product_id}|{brief.slug}".encode()).hexdigest()
                wanted = libres[int(h[:8], 16) % len(libres)]
        self._used.add(wanted)
        return wanted

    def generate_scene(self, brief: SceneBrief, width: int, height: int, seed: int) -> bytes:
        key = self._pick(brief)
        img, floor = render_studio(width, height, key, seed=seed)
        self._last[brief.slug] = (key, FloorEstimate(
            quad_norm=tuple((float(x / width), float(y / height)) for x, y in floor.quad),
            width_m=floor.width_m, depth_m=floor.depth_m, confidence=1.0,
            notes=f"plan analytique du decor studio {key}",
        ))
        self._charge("generate_scene", 0.0, key)
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            raise ProviderError("encodage du decor impossible")
        return buf.tobytes()

    def default_floor(self, width: int, height: int, brief: SceneBrief) -> FloorEstimate:
        entry = self._last.get(brief.slug)
        if entry is None:
            raise ProviderError("default_floor appele avant generate_scene")
        return entry[1]
