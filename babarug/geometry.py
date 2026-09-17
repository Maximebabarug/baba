"""Geometrie du pipeline : quadrilateres, homographies, plan du sol.

Tout ce qui garantit que le tapis garde ses proportions et se pose a plat sur
le sol passe par ici. Aucune dependance a un modele d'IA : ce fichier est
testable et deterministe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Pt = tuple[float, float]
QuadPx = np.ndarray  # (4,2) float32, ordre TL, TR, BR, BL, en pixels


def order_quad(pts: np.ndarray) -> QuadPx:
    """Ordonne 4 points en TL, TR, BR, BL.

    Tri angulaire autour du centroide, puis rotation pour demarrer sur le point
    le plus haut a gauche. Robuste a toute forme convexe.

    L'heuristique repandue -- somme x+y minimale en haut a gauche, maximale en
    bas a droite -- parait equivalente et ne l'est pas : sur un trapeze large et
    peu profond, typique d'un sol vu en perspective, le coin bas-gauche a un x
    tres negatif et une somme plus petite que celle du coin haut-gauche. Deux
    coins sont alors designes deux fois, le quad devient degenere et
    l'homographie s'effondre en silence. Constate sur un sol de 3,4 m de large
    pour 2,6 m de profondeur.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    order = np.argsort(ang)
    cyc = pts[order]

    # Depart sur le coin le plus haut (et le plus a gauche en cas d'egalite).
    start = int(np.lexsort((cyc[:, 0], cyc[:, 1]))[0])
    cyc = np.roll(cyc, -start, axis=0)

    # Sens de parcours : le second point doit etre a droite du premier.
    if cyc[1, 0] < cyc[3, 0]:
        cyc = cyc[[0, 3, 2, 1]]
    return cyc.astype(np.float32)


def quad_area(q: np.ndarray) -> float:
    """Aire par la formule du lacet (shoelace)."""
    q = np.asarray(q, dtype=np.float64).reshape(4, 2)
    x, y = q[:, 0], q[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def is_convex(q: np.ndarray) -> bool:
    """Un quad non convexe (ou croise) produit une homographie aberrante."""
    q = np.asarray(q, dtype=np.float64).reshape(4, 2)
    signs = []
    for i in range(4):
        a, b, c = q[i], q[(i + 1) % 4], q[(i + 2) % 4]
        d1, d2 = b - a, c - b
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross) < 1e-9:
            return False
        signs.append(cross > 0)
    return all(signs) or not any(signs)


def _line_intersection(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray):
    """Intersection des droites (p1p2) et (p3p4) en coordonnees homogenes."""
    l1 = np.cross(np.append(p1, 1.0), np.append(p2, 1.0))
    l2 = np.cross(np.append(p3, 1.0), np.append(p4, 1.0))
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-9:
        return None  # droites paralleles : point de fuite a l'infini, cas valide
    return p[:2] / p[2]


def floor_quad_plausibility(q: np.ndarray, img_w: int, img_h: int) -> tuple[bool, list[str]]:
    """Verifie qu'un quad peut vraiment etre une zone de sol vue en perspective.

    Un modele de vision donne des points approximatifs et se trompe regulierement.
    Ces controles geometriques attrapent les erreurs grossieres avant qu'elles ne
    deviennent un tapis flottant ou fusionne avec le canape.
    """
    problems: list[str] = []
    q = np.asarray(q, dtype=np.float64).reshape(4, 2)

    if not is_convex(q):
        problems.append("quad non convexe")

    area = quad_area(q)
    if area < 0.02 * img_w * img_h:
        problems.append(f"zone de sol trop petite ({area / (img_w * img_h):.1%} de l'image)")
    if area > 0.85 * img_w * img_h:
        problems.append("zone de sol invraisemblablement grande")

    tl, tr, br, bl = q
    top_w = np.linalg.norm(tr - tl)
    bottom_w = np.linalg.norm(br - bl)
    # Un sol vu de face/plongee : le bord proche (bas) est plus large que le bord loin.
    if bottom_w < top_w * 0.98:
        problems.append("perspective inversee : le bord proche est plus etroit que le bord lointain")
    if top_w > 0 and bottom_w / top_w > 6.0:
        problems.append("perspective extreme, angle trop rasant pour un tapis credible")

    # Le sol occupe la moitie basse : le centroide ne doit pas etre en haut du cadre.
    if q[:, 1].mean() < 0.35 * img_h:
        problems.append("zone situee trop haut dans l'image pour etre un sol")

    # Point de fuite : les cotes lateraux doivent converger vers le HAUT, pas le bas.
    vp = _line_intersection(tl, bl, tr, br)
    if vp is not None and vp[1] > q[:, 1].min():
        problems.append("les bords lateraux convergent vers le bas (geometrie impossible)")

    return (not problems), problems


@dataclass
class FloorPlane:
    """Zone de sol degagee, avec son echelle reelle.

    Sans echelle, on ne peut pas poser un tapis de 200x140 cm : on ne saurait
    que l'inscrire "joliment" dans la zone, ce qui produit des tapis de 4 metres
    ou des paillassons. width_m / depth_m sont estimes par le modele de vision a
    partir du mobilier visible (un canape fait ~2 m, une marche ~30 cm) puis
    bornes a des valeurs plausibles. L'estimation est approximative et c'est
    assume : elle n'a besoin d'etre bonne qu'a ~15 % pour que le tapis soit
    credible.
    """

    quad: QuadPx                 # (4,2) px, ordre TL, TR, BR, BL
    width_m: float = 4.0         # largeur reelle du bord LOINTAIN au bord proche
    depth_m: float = 3.0

    def __post_init__(self):
        self.quad = order_quad(np.asarray(self.quad, dtype=np.float32))
        self.width_m = float(np.clip(self.width_m, 1.2, 12.0))
        self.depth_m = float(np.clip(self.depth_m, 1.0, 12.0))

    @property
    def homography(self):
        """unite [0,1]^2 -> pixels de l'image."""
        import cv2

        unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        return cv2.getPerspectiveTransform(unit, self.quad)

    def metric_to_unit(self, x_m: float, y_m: float) -> tuple[float, float]:
        return x_m / self.width_m, y_m / self.depth_m


def fit_rug_quad(
    floor: "FloorPlane | np.ndarray",
    aspect_ratio: float,
    real_size_cm: tuple[float, float] | None = None,
    fill: float = 0.82,
    center_u: float = 0.5,
    center_v: float = 0.5,
    rotation_deg: float = 0.0,
) -> QuadPx:
    """Place le tapis sur le plan du sol, a sa taille reelle, en perspective.

    On raisonne en METRES dans le plan du sol, puis on projette. C'est ce qui
    garantit qu'un tapis de 300x200 est visiblement plus grand qu'un 150x100
    dans la meme piece -- et que ses proportions restent justes quel que soit
    l'angle de la camera.

    real_size_cm : (longueur, largeur). Si absent, on retombe sur une taille
    deduite du ratio et d'une surface de reference (2.6 m2), en le signalant
    par la valeur de retour identique -- le QC mesurera le ratio, pas la taille.
    La longueur est posee selon l'axe de profondeur de la piece.
    """
    import cv2

    if not isinstance(floor, FloorPlane):
        floor = FloorPlane(np.asarray(floor, dtype=np.float32))

    if real_size_cm:
        length_m, width_m = real_size_cm[0] / 100.0, real_size_cm[1] / 100.0
    else:
        area = 2.6  # m2, tapis de salon courant
        width_m = float(np.sqrt(area / max(aspect_ratio, 1e-6)))
        length_m = width_m * aspect_ratio

    # Le tapis ne doit pas deborder de la zone degagee.
    max_w = floor.width_m * fill
    max_d = floor.depth_m * fill
    shrink = min(1.0, max_w / max(width_m, 1e-6), max_d / max(length_m, 1e-6))
    width_m *= shrink
    length_m *= shrink

    hu, hv = floor.metric_to_unit(width_m / 2.0, length_m / 2.0)
    local = np.array(
        [[-hu, -hv], [hu, -hv], [hu, hv], [-hu, hv]], dtype=np.float32
    )
    if abs(rotation_deg) > 1e-6:
        # Rotation dans le plan du sol, en metres, pour rester isotrope.
        t = np.radians(rotation_deg)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]], dtype=np.float32)
        m = local * np.array([floor.width_m, floor.depth_m], np.float32)
        m = m @ R.T
        local = m / np.array([floor.width_m, floor.depth_m], np.float32)

    cu = float(np.clip(center_u, abs(local[:, 0]).max(), 1 - abs(local[:, 0]).max()))
    cv_ = float(np.clip(center_v, abs(local[:, 1]).max(), 1 - abs(local[:, 1]).max()))
    local = local + np.array([cu, cv_], dtype=np.float32)

    projected = cv2.perspectiveTransform(
        local.reshape(1, 4, 2).astype(np.float32), floor.homography
    ).reshape(4, 2)
    return order_quad(projected.astype(np.float32))


def rug_quad_metrics(rug_quad: np.ndarray, floor: "FloorPlane") -> tuple[float, float]:
    """Dimensions reelles (longueur_m, largeur_m) d'un quad pose sur le sol.

    Rectification metrique : on ramene le quad dans le plan du sol puis on
    remet l'echelle. C'est la mesure qui permet de verifier apres coup que les
    proportions n'ont pas derive.
    """
    import cv2

    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    Hinv = cv2.getPerspectiveTransform(floor.quad, unit)
    q = cv2.perspectiveTransform(
        order_quad(np.asarray(rug_quad, np.float32)).reshape(1, 4, 2), Hinv
    ).reshape(4, 2)
    metric = q * np.array([floor.width_m, floor.depth_m], np.float32)
    tl, tr, br, bl = metric.astype(np.float64)
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    d = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    return float(max(w, d)), float(min(w, d))


def denormalize(quad_norm, w: int, h: int) -> QuadPx:
    q = np.asarray(quad_norm, dtype=np.float32).reshape(4, 2)
    return np.column_stack([q[:, 0] * w, q[:, 1] * h]).astype(np.float32)


def normalize(quad_px: np.ndarray, w: int, h: int):
    q = np.asarray(quad_px, dtype=np.float32).reshape(4, 2)
    return tuple((float(x / w), float(y / h)) for x, y in q)
