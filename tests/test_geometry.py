"""Le tapis doit garder ses proportions reelles quel que soit l'angle."""
import numpy as np
import pytest

from babarug.geometry import (
    FloorPlane, fit_rug_quad, floor_quad_plausibility, is_convex,
    order_quad, quad_area, rug_quad_metrics,
)


def test_order_quad_est_idempotent():
    q = np.float32([[10, 10], [90, 12], [95, 60], [8, 58]])
    assert np.allclose(order_quad(q), order_quad(order_quad(q)))


def test_sol_plausible_accepte_une_perspective_normale():
    ok, pb = floor_quad_plausibility(np.float32([[300, 500], [900, 500], [1100, 850], [100, 850]]), 1200, 900)
    assert ok, pb


def test_sol_rejette_une_perspective_inversee():
    """Bord proche plus etroit que le bord lointain : geometriquement impossible."""
    ok, pb = floor_quad_plausibility(np.float32([[100, 500], [1100, 500], [900, 850], [300, 850]]), 1200, 900)
    assert not ok and any("inversee" in p for p in pb)


def test_sol_rejette_une_zone_trop_haute():
    ok, pb = floor_quad_plausibility(np.float32([[300, 50], [900, 50], [1000, 200], [200, 200]]), 1200, 900)
    assert not ok


@pytest.mark.parametrize("length,width", [(300, 200), (200, 140), (250, 80), (150, 150)])
def test_les_dimensions_reelles_sont_restituees(length, width):
    """Coeur de la fidelite geometrique : un 250x80 doit rester un 250x80."""
    floor = FloorPlane(np.float32([[480, 456], [1120, 456], [1472, 873], [128, 873]]), 4.2, 3.4)
    q = fit_rug_quad(floor, length / width, real_size_cm=(length, width), fill=0.95)
    L, W = rug_quad_metrics(q, floor)
    assert L / W == pytest.approx(length / width, rel=0.02)
    assert L * 100 == pytest.approx(length, rel=0.05)


def test_le_tapis_reste_dans_la_zone_de_sol():
    floor = FloorPlane(np.float32([[480, 456], [1120, 456], [1472, 873], [128, 873]]), 4.2, 3.4)
    q = fit_rug_quad(floor, 1.5, real_size_cm=(900, 600))  # tapis absurdement grand
    assert quad_area(q) <= quad_area(floor.quad)
    assert is_convex(q)
