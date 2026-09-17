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


def zoom_scene(
    scene_bgr: np.ndarray, floor: FloorPlane, focus: np.ndarray, factor: float
) -> tuple[np.ndarray, FloorPlane]:
    """Rapproche la camera : recadre la scene autour du tapis, puis reechantillonne.

    C'est le bon geste quand le tapis est fidele mais trop petit dans l'image.
    Agrandir le tapis serait mentir sur le produit : un 170x133 doit occuper la
    place d'un 170x133 dans la piece. Un photographe, lui, se rapproche -- c'est
    exactement ce que fait cette fonction.

    Le quad du sol est transporte dans le nouveau repere : l'oublier ferait poser
    le tapis a cote de la zone prevue.
    """
    h, w = scene_bgr.shape[:2]
    factor = float(np.clip(factor, 1.0, 3.0))
    cw, ch = w / factor, h / factor

    cx, cy = np.asarray(focus, dtype=np.float64).reshape(4, 2).mean(axis=0)
    x0 = float(np.clip(cx - cw / 2, 0, w - cw))
    # On laisse un peu plus d'air au-dessus du tapis qu'en dessous : le regard
    # doit voir le mobilier derriere, pas seulement du sol.
    y0 = float(np.clip(cy - ch * 0.58, 0, h - ch))

    crop = scene_bgr[int(y0):int(y0 + ch), int(x0):int(x0 + cw)]
    out = cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)

    q = np.asarray(floor.quad, dtype=np.float32).reshape(4, 2)
    moved = np.column_stack([(q[:, 0] - x0) * (w / cw), (q[:, 1] - y0) * (h / ch)])
    return out, FloorPlane(moved.astype(np.float32), floor.width_m, floor.depth_m)
