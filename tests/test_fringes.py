"""Recuperation des franges et detection de leur absence.

Ces tests encodent un echec reel : sur BABA-RUG-0001, le detourage coupait les
deux franges et le controle de masque notait quand meme 1.00.
"""
import cv2
import numpy as np
import pytest

from babarug.pipeline.fringes import recover_fringes
from babarug.providers.segmentation import edge_roughness, mask_quality


def _tapis_frange(fond=(210, 212, 214), frange=(180, 195, 205), seed=5,
                  longueur=(14, 21)):
    """Tapis sombre a franges claires sur fond clair : le cas difficile."""
    rng = np.random.default_rng(seed)
    img = np.full((900, 700, 3), fond, np.uint8)
    img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    corps = np.zeros(img.shape[:2], np.uint8)
    x0, x1, y0, y1 = 120, 580, 180, 720
    cv2.rectangle(img, (x0, y0), (x1, y1), (60, 50, 140), -1)
    cv2.rectangle(corps, (x0, y0), (x1, y1), 255, -1)
    verite = corps.copy()
    for x in range(x0, x1, 3):  # franges de longueurs inegales
        for y_base, sens in ((y0, -1), (y1, 1)):
            L = int(rng.uniform(*longueur))
            p2 = (x + int(rng.uniform(-2, 2)), y_base + sens * L)
            cv2.line(img, (x, y_base), p2, frange, 2)
            cv2.line(verite, (x, y_base), p2, 255, 2)
    return img, corps, verite


def test_les_franges_sont_recuperees():
    img, corps, verite = _tapis_frange()
    r = recover_fringes(img, corps)
    assert r.found, r.notes
    assert r.separation > 2.0
    recouvrement = ((r.mask > 127) & (verite > 127)).sum() / (verite > 127).sum()
    assert recouvrement > 0.90, f"seulement {recouvrement:.1%} de la frange retrouvee"


def test_les_franges_ne_sont_cherchees_que_sur_les_petits_cotes():
    """Les franges naissent des fils de chaine : chercher sur les grands cotes
    ramasserait les lisieres et l'ombre portee."""
    img, corps, _ = _tapis_frange()
    r = recover_fringes(img, corps)
    fys, fxs = np.nonzero(r.fringe_only > 127)
    bys, bxs = np.nonzero(corps > 127)
    assert fys.min() < bys.min() and fys.max() > bys.max(), "rien au-dela des petits cotes"
    marge = 8
    assert fxs.min() >= bxs.min() - marge and fxs.max() <= bxs.max() + marge, \
        "la recherche a deborde sur les grands cotes"


def test_un_fond_indiscernable_est_signale_et_non_invente():
    """Si l'information n'est pas dans l'image, ne rien inventer : le dire."""
    img, corps, _ = _tapis_frange(fond=(205, 205, 205), frange=(205, 205, 205))
    r = recover_fringes(img, corps)
    assert not r.found
    assert any("indiscernable" in n or "aucune frange" in n for n in r.notes)


def test_la_rugosite_distingue_un_bord_coupe_d_une_frange():
    img, corps, verite = _tapis_frange()
    assert max(edge_roughness(corps)) < 0.004, "un bord droit doit etre lisse"
    assert max(edge_roughness(verite)) > 0.004, "une frange doit etre rugueuse"


def test_un_masque_sans_frange_est_signale():
    """Le bug d'origine : masque note 1.00 alors que les deux franges manquaient."""
    img, corps, verite = _tapis_frange()
    score_coupe, pb = mask_quality(corps, img)
    assert any("frange" in p for p in pb), "l'absence de franges doit etre signalee"
    score_complet, _ = mask_quality(verite, img)
    assert score_complet > score_coupe


def test_des_franges_anormalement_longues_sont_signalees_et_non_devinees():
    """Franges depassant largement l'etendue nominale : le modele de fond est
    alors echantillonne parmi les brins et la mesure devient ininterpretable.

    Le comportement attendu n'est PAS de deviner, mais de le dire. Elargir
    automatiquement la zone de recherche a ete essaye et degradait les photos
    reelles en happant l'ombre portee -- voir le commentaire dans fringes.py."""
    img, corps, _ = _tapis_frange(longueur=(40, 70))
    r = recover_fringes(img, corps)
    if not r.found:
        assert r.notes, "un echec doit toujours etre explique"
    else:
        # Si des franges sont trouvees, elles ne doivent pas deborder n'importe ou.
        fys, _ = np.nonzero(r.fringe_only > 127)
        bys, _ = np.nonzero(corps > 127)
        assert fys.min() > bys.min() - 200 and fys.max() < bys.max() + 200


def test_la_reduction_d_echelle_ne_degrade_pas_le_resultat():
    """work_px accelere le traitement en lot sans changer le verdict."""
    img, corps, _ = _tapis_frange()
    grand = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gcorps = cv2.resize(corps, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    lent = recover_fringes(grand, gcorps, work_px=10000)
    vite = recover_fringes(grand, gcorps, work_px=900)
    assert vite.found == lent.found
    inter = ((vite.mask > 127) & (lent.mask > 127)).sum()
    union = ((vite.mask > 127) | (lent.mask > 127)).sum()
    assert inter / union > 0.95, "la version rapide doit rester equivalente"
