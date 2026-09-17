"""La plate doit redresser le tapis sans le deformer ni raboter ses franges."""
import cv2
import numpy as np
import pytest

from babarug.pipeline.plate import extract_plate, measure_aspect_ratio


def test_la_plate_respecte_le_ratio_demande(plate):
    w, h = plate.size
    assert max(w, h) / min(w, h) == pytest.approx(plate.aspect_ratio, rel=0.02)


def test_la_plate_a_un_canal_alpha_utile(plate):
    assert plate.rgba.shape[2] == 4
    couverture = (plate.rgba[:, :, 3] > 127).mean()
    assert 0.80 < couverture < 1.0, "le tapis doit remplir la plate sans la saturer"


def test_les_franges_survivent_a_la_rectification(plate):
    """Les franges sont des excroissances fines : elles doivent rester dans l'alpha."""
    a = plate.rgba[:, :, 3]
    h = a.shape[0]
    bande_haute = a[: int(h * 0.04)]
    assert (bande_haute > 127).any(), "la bande de frange a ete perdue"


def test_le_ratio_incoherent_est_signale(rug_photo):
    photo, mask = rug_photo
    p = extract_plate(photo, mask, aspect_ratio=6.0)  # ratio absurde
    assert any("ratio DNA" in w for w in p.warnings)


def test_un_masque_vide_leve_une_erreur(rug_photo):
    photo, _ = rug_photo
    with pytest.raises(ValueError):
        extract_plate(photo, np.zeros(photo.shape[:2], np.uint8), 1.5)
