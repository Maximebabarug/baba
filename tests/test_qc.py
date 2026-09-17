"""Le controle qualite doit ATTRAPER les alterations du produit.

C'est la suite de tests la plus importante du projet : elle repond a la question
"le systeme sait-il refuser une belle image montrant un autre tapis ?".
Chaque cas ci-dessous est un mode d'echec cite au cahier des charges.
"""
import cv2
import numpy as np
import pytest

from babarug.models import SceneReview, Verdict
from babarug.pipeline.composite import CompositeOptions, composite_rug
from babarug.pipeline.plate import Plate
from babarug.pipeline.qc import decide, measure_fidelity
from babarug.geometry import fit_rug_quad
from synth import synth_rug


def _rendre(room, floor, vraie_plate, plate_composee=None, quad=None):
    """Compose `plate_composee` mais MESURE toujours contre `vraie_plate`.

    Le tapis est cadre genereusement : une image de fiche produit doit montrer
    le tapis, et le controle qualite signale a juste titre un cadrage ou il
    n'occupe qu'une poignee de pour cent de l'image."""
    quad = quad if quad is not None else fit_rug_quad(
        floor, vraie_plate.aspect_ratio, real_size_cm=(300, 193), fill=0.98
    )
    res = composite_rug(room, plate_composee or vraie_plate, quad, CompositeOptions())
    m, notes = measure_fidelity(
        vraie_plate, res.image, res.H_plate_to_scene, quad, floor, res.rug_alpha
    )
    return decide("T", m, None, notes)


def _variante(plate, bgr=None, alpha=None):
    return Plate(
        np.dstack([bgr if bgr is not None else plate.rgba[:, :, :3],
                   alpha if alpha is not None else plate.rgba[:, :, 3]]),
        plate.source_corners, plate.aspect_ratio, 0,
    )


def test_le_bon_tapis_est_approuve(plate, room):
    img, floor = room
    r = _rendre(img, floor, plate)
    assert r.verdict == Verdict.APPROVED, r.failures
    assert r.fidelity.delta_e_mean < 3.0


def test_un_autre_tapis_est_rejete(plate, room):
    """Le scenario redoute : belle image, mauvais produit."""
    img, floor = room
    autre = cv2.resize(synth_rug(seed=99), plate.size)
    r = _rendre(img, floor, plate, _variante(plate, bgr=autre))
    assert r.verdict == Verdict.REJECTED


def test_un_motif_central_modifie_est_rejete(plate, room):
    img, floor = room
    tam = plate.rgba.copy()
    h, w = tam.shape[:2]
    cv2.ellipse(tam, (w // 2, h // 2), (150, 260), 0, 0, 360, (40, 60, 150), -1)
    cv2.rectangle(tam, (w // 2 - 90, h // 2 - 150), (w // 2 + 90, h // 2 + 150), (200, 205, 215), 14)
    r = _rendre(img, floor, plate, Plate(tam, plate.source_corners, plate.aspect_ratio, 0))
    assert r.verdict == Verdict.REJECTED


def test_des_couleurs_embellies_sont_rejetees(plate, room):
    """'Ne change pas ses couleurs pour qu'elles correspondent au decor.'"""
    img, floor = room
    hsv = cv2.cvtColor(plate.rgba[:, :, :3], cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= 1.35
    hsv[..., 0] = (hsv[..., 0] - 6) % 180
    warm = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    r = _rendre(img, floor, plate, _variante(plate, bgr=warm))
    assert r.verdict == Verdict.REJECTED


def test_une_texture_lissee_est_detectee(plate, room):
    """'Ne rends pas le tapis plus parfait.' -- usure et grain effaces."""
    img, floor = room
    lisse = cv2.medianBlur(plate.rgba[:, :, :3], 21)
    r = _rendre(img, floor, plate, _variante(plate, bgr=lisse))
    assert r.verdict != Verdict.APPROVED
    assert r.fidelity.texture_retention < 0.80


def test_des_franges_supprimees_sont_detectees(plate, room):
    """'Ne pas appliquer un detourage artificiellement parfait.'"""
    img, floor = room
    rogne = cv2.erode(plate.rgba[:, :, 3], cv2.getStructuringElement(cv2.MORPH_RECT, (3, 41)))
    r = _rendre(img, floor, plate, _variante(plate, alpha=rogne))
    assert r.verdict != Verdict.APPROVED
    assert r.fidelity.silhouette_iou < 0.975


def test_des_proportions_deformees_sont_rejetees(plate, room):
    img, floor = room
    carre = fit_rug_quad(floor, 1.0, real_size_cm=(190, 190))
    r = _rendre(img, floor, plate, quad=carre)
    assert r.verdict == Verdict.REJECTED
    assert r.fidelity.scale_error > 0.12


def test_un_tapis_hors_cadre_est_rejete(plate, room):
    img, floor = room
    q = fit_rug_quad(floor, plate.aspect_ratio, real_size_cm=(300, 193), fill=0.98) \
        + np.float32([[0, 330]] * 4)
    r = _rendre(img, floor, plate, quad=q)
    assert r.verdict == Verdict.REJECTED
    assert r.fidelity.visible_fraction < 0.90


def test_un_avis_de_scene_ne_peut_pas_repecher_un_echec_de_fidelite(plate, room):
    """Regle cardinale : un score d'IA n'est jamais une preuve."""
    img, floor = room
    autre = cv2.resize(synth_rug(seed=99), plate.size)
    quad = fit_rug_quad(floor, plate.aspect_ratio, real_size_cm=(300, 193), fill=0.98)
    res = composite_rug(img, _variante(plate, bgr=autre), quad, CompositeOptions())
    m, notes = measure_fidelity(plate, res.image, res.H_plate_to_scene, quad, floor, res.rug_alpha)
    parfait = SceneReview(scene_realism=100, perspective=100, composition=100, integration=100,
                          rug_fully_visible=True, rug_merges_with_furniture=False)
    assert decide("T", m, parfait, notes).verdict == Verdict.REJECTED


def test_un_avis_de_scene_peut_degrader_une_image_fidele(plate, room):
    img, floor = room
    quad = fit_rug_quad(floor, plate.aspect_ratio, real_size_cm=(300, 193), fill=0.98)
    res = composite_rug(img, plate, quad, CompositeOptions())
    m, notes = measure_fidelity(plate, res.image, res.H_plate_to_scene, quad, floor, res.rug_alpha)
    mauvais = SceneReview(scene_realism=90, perspective=90, composition=90, integration=90,
                          rug_fully_visible=True, rug_merges_with_furniture=True)
    assert decide("T", m, mauvais, notes).verdict == Verdict.REJECTED
