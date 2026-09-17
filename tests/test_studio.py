"""Studio procedural et regression du tri des coins."""
import cv2
import numpy as np
import pytest

from babarug.geometry import fit_rug_quad, is_convex, order_quad, quad_area
from babarug.pipeline.composite import CompositeOptions, composite_rug
from babarug.pipeline.plate import Plate
from babarug.pipeline.qc import decide, measure_fidelity
from babarug.studio import STUDIO_STYLES, Camera, render_studio


# ------------------------------------------------------- regression order_quad
def test_le_tri_des_coins_tient_sur_un_trapeze_large():
    """Regression : un sol vu en perspective est large et peu profond.

    L'ancienne heuristique somme/difference designait deux fois les memes coins
    (le bas-gauche ayant un x tres negatif, sa somme passait sous celle du
    haut-gauche). Le quad devenait degenere, l'homographie s'effondrait, et le
    tapis disparaissait de l'image SANS message d'erreur.
    """
    q = order_quad(np.float32([[306, 528], [1294, 528], [1677, 879], [-77, 879]]))
    assert len({tuple(p) for p in np.round(q).astype(int).tolist()}) == 4
    assert is_convex(q)
    assert q[0][1] < q[3][1] and q[0][0] < q[1][0]   # TL au-dessus de BL, a gauche de TR


@pytest.mark.parametrize("perm", [[0, 1, 2, 3], [2, 3, 0, 1], [3, 1, 0, 2], [1, 0, 3, 2]])
def test_le_tri_des_coins_est_invariant_a_l_ordre_d_entree(perm):
    base = np.float32([[306, 528], [1294, 528], [1677, 879], [-77, 879]])
    assert np.allclose(order_quad(base), order_quad(base[perm]))


# ------------------------------------------------------- camera
def test_la_camera_projette_le_sol_de_facon_monotone():
    """Plus un point du sol est loin, plus il est haut dans l'image."""
    cam = Camera(height_m=1.5, pitch_deg=14.0, focal_mm=38.0)
    p = cam.projector(1600, 900)
    vs = [float(p(0.0, Y)[1]) for Y in (3.0, 4.0, 5.0, 6.0)]
    assert vs == sorted(vs, reverse=True)


def test_la_zone_degagee_est_dans_le_cadre():
    """Elle etait posee au juge et tombait sous le bord inferieur de l'image."""
    for key in STUDIO_STYLES:
        img, floor = render_studio(1200, 675, key, seed=2)
        q = floor.quad
        assert (q[:, 1] < 675).all(), f"{key} : zone sous le bas du cadre"
        assert (q[:, 1] > 0).all(), f"{key} : zone au-dessus du cadre"
        assert is_convex(q) and quad_area(q) > 0.05 * 1200 * 675


# ------------------------------------------------------- rendu
@pytest.mark.parametrize("key", list(STUDIO_STYLES))
def test_chaque_decor_se_rend_et_a_de_la_matiere(key):
    img, floor = render_studio(1200, 675, key, seed=1)
    assert img.shape == (675, 1200, 3)
    # Un aplat trahit le rendu : on exige de la variation locale dans le sol.
    bas = cv2.cvtColor(img[450:, :], cv2.COLOR_BGR2GRAY).astype(np.float32)
    assert bas.std() > 6.0, "le sol est trop uniforme pour passer pour une matiere"
    assert floor.width_m > 1.0 and floor.depth_m > 0.5


def test_le_rendu_est_reproductible():
    a, _ = render_studio(800, 450, "parisien_chene", seed=7)
    b, _ = render_studio(800, 450, "parisien_chene", seed=7)
    assert np.array_equal(a, b)
    c, _ = render_studio(800, 450, "parisien_chene", seed=8)
    assert not np.array_equal(a, c)


# ------------------------------------------------------- bout en bout
def test_le_tapis_pose_dans_le_studio_reste_fidele(plate):
    img, floor = render_studio(1600, 900, "parisien_chene", seed=5)
    q = fit_rug_quad(floor, plate.aspect_ratio, real_size_cm=(230, 148), fill=0.95)
    res = composite_rug(img, plate, q, CompositeOptions())
    m, n = measure_fidelity(plate, res.image, res.H_plate_to_scene, q, floor, res.rug_alpha)
    rep = decide("S", m, None, n)
    assert m.delta_e_mean < 2.0, rep.failures
    assert m.ssim > 0.97 and m.visible_fraction > 0.99
    assert m.texture_retention > 0.85
