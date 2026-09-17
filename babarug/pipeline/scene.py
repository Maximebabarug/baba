"""Etape 4 : la SCENE.

Genere la piece VIDE puis y localise le plan du sol. Le tapis n'est jamais
demande au generateur : c'est la regle qui fait tenir tout l'edifice.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from babarug.geometry import FloorPlane, denormalize, floor_quad_plausibility
from babarug.models import SceneBrief
from babarug.providers.base import FloorEstimate, ProviderError

log = logging.getLogger(__name__)


def _fallback_floor(w: int, h: int, brief: SceneBrief) -> FloorEstimate:
    """Sol par defaut quand la lecture echoue : trapeze plausible en bas du cadre.

    Moins bon qu'une vraie lecture, mais preferable a un abandon : l'operateur
    verra le resultat et pourra corriger le quad a la main dans l'interface.
    """
    y0 = 0.52 + 0.04 * (brief.camera_pitch_deg / -20.0)
    y0 = float(np.clip(y0, 0.45, 0.62))
    return FloorEstimate(
        quad_norm=((0.30, y0), (0.70, y0), (0.93, 0.97), (0.07, 0.97)),
        width_m=4.0, depth_m=3.2, confidence=0.25,
        notes="sol estime par defaut : la lecture du plan a echoue",
    )


def generate_scene(
    gen_provider, vision_provider, brief: SceneBrief, width: int, height: int, seed: int
) -> tuple[np.ndarray, FloorPlane, list[str]]:
    """-> (image BGR de la piece vide, plan du sol, avertissements)."""
    warnings: list[str] = []
    png = gen_provider.generate_scene(brief, width, height, seed)
    scene = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    if scene is None:
        raise ProviderError("la scene generee est illisible")

    # Le generateur procedural connait son propre sol : inutile de payer une
    # lecture de vision pour une information deja exacte.
    est: FloorEstimate | None = None
    if hasattr(gen_provider, "default_floor"):
        est = gen_provider.default_floor(width, height, brief)
    else:
        try:
            est = vision_provider.locate_floor(png, brief)
        except ProviderError as e:
            warnings.append(f"lecture du sol impossible ({e}) : repli sur un sol par defaut")

    if est is None:
        est = _fallback_floor(width, height, brief)

    quad_px = denormalize(est.quad_norm, width, height)
    ok, problems = floor_quad_plausibility(quad_px, width, height)
    if not ok:
        warnings.append(
            "plan du sol invraisemblable (" + " ; ".join(problems) + ") : repli sur un sol par defaut"
        )
        est = _fallback_floor(width, height, brief)
        quad_px = denormalize(est.quad_norm, width, height)
    if est.confidence < 0.5:
        warnings.append(f"confiance faible sur le plan du sol ({est.confidence:.2f})")

    return scene, FloorPlane(quad_px, est.width_m, est.depth_m), warnings
