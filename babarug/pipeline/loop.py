"""Orchestration : GENERER -> ANALYSER -> COMPARER -> CORRIGER -> RE-ANALYSER.

La boucle ne "retente au hasard" pas : chaque echec de QC est traduit en une
correction CIBLEE (recadrer, repositionner, baisser le relighting, regenerer la
piece). Relancer a l'identique en esperant mieux coute de l'argent et ne corrige
rien. Le nombre d'iterations est borne.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from babarug.geometry import FloorPlane, fit_rug_quad
from babarug.models import QCReport, RenderResult, RugDNA, SceneBrief, Verdict
from babarug.pipeline.composite import CompositeOptions, composite_rug
from babarug.pipeline.plate import Plate
from babarug.pipeline.qc import QCThresholds, decide, measure_fidelity
from babarug.pipeline.scene import generate_scene, zoom_scene

log = logging.getLogger(__name__)


@dataclass
class RenderSettings:
    width: int = 1600
    height: int = 900
    max_iterations: int = 3
    fill: float = 0.80
    center_v: float = 0.56          # le tapis part de l'assise vers la camera
    review_with_model: bool = True
    thresholds: QCThresholds = field(default_factory=QCThresholds)
    composite: CompositeOptions = field(default_factory=CompositeOptions)


@dataclass
class _State:
    """Ce que la boucle a le droit de modifier entre deux iterations."""

    fill: float
    center_v: float
    seed: int
    relight: float
    zoom: float = 1.0
    regenerate_scene: bool = False
    reason: str = ""


def _plan_correction(report: QCReport, st: _State, settings: RenderSettings) -> _State | None:
    """Traduit un verdict en action. Retourne None si rien d'utile a tenter.

    Ordre des priorites : d'abord ce qui se corrige sans repayer une generation
    (cadrage, position, relighting), ensuite seulement la regeneration de la
    piece, qui est l'operation la plus couteuse.
    """
    f = " ".join(report.failures).lower()
    new = _State(st.fill, st.center_v, st.seed, st.relight)

    if "cadrage trop large" in f or "n'occupe que" in f:
        # Le tapis est fidele mais trop petit dans l'image. On RAPPROCHE LA
        # CAMERA. Augmenter son emprise au sol serait inutile et malhonnete :
        # `fit_rug_quad` ne fait que brider un tapis trop grand pour la zone, il
        # ne l'agrandit jamais au-dela de ses dimensions reelles. Une premiere
        # version poussait `fill` a chaque iteration et ne changeait donc
        # strictement rien -- trois tours de boucle pour rien.
        if st.zoom >= 2.6:
            return None
        new.zoom = min(2.6, st.zoom * 1.35)
        new.reason = (f"tapis trop petit dans l'image : on rapproche la camera "
                      f"(zoom x{new.zoom:.2f})")
        return new
    if "hors cadre" in f or "entierement visible" in f:
        new.fill = max(0.55, st.fill - 0.12)
        new.center_v = float(np.clip(st.center_v - 0.06, 0.35, 0.75))
        new.reason = "tapis coupe : on reduit l'emprise et on le remonte dans le cadre"
        return new
    if "proportions" in f:
        new.reason = "proportions deviees : on recalcule le quad sur le plan du sol"
        return new
    if "relighting" in f or "delave" in f:
        new.relight = max(0.25, st.relight - 0.30)
        new.reason = "relighting trop fort : on reduit son influence sur le tapis"
        return new
    if "texture" in f:
        new.relight = max(0.3, st.relight - 0.15)
        new.reason = "texture attenuee : on allege les traitements appliques au tapis"
        return new
    # Attention : comparer sur des expressions entieres. Un test sur "sol" seul
    # matcherait "resolution" et declencherait des regenerations inutiles -- et
    # facturees. Bug reellement rencontre.
    if any(k in f for k in ("fusionne", "perspective", "realisme de scene",
                            "plan du sol", "invraisemblable")):
        new.regenerate_scene = True
        new.seed = st.seed + 977
        new.reason = "probleme de scene : on regenere la piece avec une autre graine"
        return new
    if "derive colorimetrique" in f:
        new.relight = max(0.2, st.relight - 0.35)
        new.reason = "derive de couleur : on neutralise davantage l'eclairage sur le tapis"
        return new
    return None


def render_variant(
    dna: RugDNA,
    plate: Plate,
    brief: SceneBrief,
    variant: str,
    gen_provider,
    vision_provider,
    qc_provider=None,
    settings: RenderSettings | None = None,
    seed: int = 1,
    on_step=None,
) -> tuple[RenderResult, np.ndarray, list[str]]:
    """Produit UNE image lifestyle, avec sa boucle de correction.

    -> (resultat, image BGR finale, avertissements)
    """
    s = settings or RenderSettings()
    t0 = time.time()
    warnings: list[str] = []
    st = _State(s.fill, s.center_v, seed, s.composite.relight_strength)

    scene = floor = None
    best: tuple[QCReport, np.ndarray, np.ndarray] | None = None

    for it in range(s.max_iterations):
        if on_step:
            on_step(f"{variant} : iteration {it + 1}/{s.max_iterations}")

        if scene is None or st.regenerate_scene:
            scene, floor, w = generate_scene(
                gen_provider, vision_provider, brief, s.width, s.height, st.seed
            )
            warnings += w
            st.regenerate_scene = False

        view, view_floor = scene, floor
        if st.zoom > 1.001:
            provisoire = fit_rug_quad(
                floor, plate.aspect_ratio, real_size_cm=dna.real_size_cm,
                fill=st.fill, center_v=st.center_v,
            )
            view, view_floor = zoom_scene(scene, floor, provisoire, st.zoom)

        quad = fit_rug_quad(
            view_floor, plate.aspect_ratio, real_size_cm=dna.real_size_cm,
            fill=st.fill, center_v=st.center_v,
        )
        opts = CompositeOptions(**{**s.composite.__dict__, "relight_strength": st.relight})
        comp = composite_rug(view, plate, quad, opts)

        metrics, notes = measure_fidelity(
            plate, comp.image, comp.H_plate_to_scene, quad, view_floor,
            comp.rug_alpha, s.thresholds
        )

        review = None
        # L'avis de scene coute un appel : on ne le demande que si la fidelite
        # mesuree tient deja. Inutile de faire juger l'esthetique d'une image
        # deja disqualifiee sur le produit.
        if s.review_with_model and qc_provider is not None and metrics.delta_e_mean < \
                s.thresholds.delta_e_mean_reject:
            try:
                png = cv2.imencode(".png", comp.image)[1].tobytes()
                review = qc_provider.review_scene(png, brief, dna)
            except Exception as e:  # un avis consultatif ne doit jamais casser le run
                warnings.append(f"revue de scene indisponible : {e}")

        report = decide(variant, metrics, review, notes, s.thresholds, iteration=it)
        log.info("%s it=%d %s -> %s", variant, it, metrics.summary, report.verdict.value)

        if best is None or _rank(report) > _rank(best[0]):
            best = (report, comp.image, comp.rug_alpha)

        if report.verdict == Verdict.APPROVED:
            break

        plan = _plan_correction(report, st, s)
        if plan is None:
            warnings.append(
                f"{variant} : aucune correction automatique ne repond a "
                f"« {'; '.join(report.failures[:2])} » -- intervention humaine requise"
            )
            break
        warnings.append(f"{variant} it{it + 1} -> correction : {plan.reason}")
        st = plan

    assert best is not None
    report, image, _ = best
    cost = sum(
        c.usd
        for p in (gen_provider, vision_provider, qc_provider)
        if p is not None
        for c in getattr(p, "costs", [])
    )
    result = RenderResult(
        variant=variant, image_path="", brief=brief, qc=report,
        iterations=report.iteration + 1, cost_usd=round(cost, 4),
        seconds=round(time.time() - t0, 1),
    )
    return result, image, warnings


def _rank(r: QCReport) -> tuple:
    """Classe deux tentatives : verdict d'abord, puis fidelite mesuree."""
    order = {Verdict.APPROVED: 2, Verdict.REVIEW_REQUIRED: 1, Verdict.REJECTED: 0}
    m = r.fidelity
    return (order[r.verdict], -m.delta_e_mean, m.ssim, m.visible_fraction)
