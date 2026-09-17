"""Etape 3 : la PLATE.

La plate est une orthophoto du tapis reel : le tapis redresse a plat, au bon
ratio, en RGBA avec ses franges. C'est l'objet qui sera ensuite RE-PROJETE dans
la scene. Les pixels du tapis final viennent d'ici, pas d'un generateur : c'est
ce qui rend la fidelite structurelle et non probabiliste.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from babarug.geometry import order_quad


@dataclass
class Plate:
    rgba: np.ndarray               # (H,W,4) uint8, tapis redresse, fond transparent
    source_corners: np.ndarray     # (4,2) coins detectes dans la photo d'origine
    aspect_ratio: float            # ratio effectivement utilise
    fringe_pad_px: int             # marge conservee pour les franges
    warnings: list[str] = field(default_factory=list)

    @property
    def size(self) -> tuple[int, int]:
        return self.rgba.shape[1], self.rgba.shape[0]


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Garde la plus grosse zone connexe : elimine les faux positifs du detourage."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8), 8)
    if n <= 1:
        return mask
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == biggest, 255, 0).astype(np.uint8)


def _corners_from_mask(mask: np.ndarray, erode_px: int) -> tuple[np.ndarray, list[str]]:
    """Coins du CHAMP tisse, franges exclues.

    Les franges sont filiformes et bruitees : si on cherche les coins sur le
    masque complet, l'homographie part de travers. On erode pour ne garder que
    le corps du tapis, on mesure les coins dessus, et la marge frange est
    reajoutee ensuite (`_expand_quad`).
    """
    warnings: list[str] = []
    body = mask
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1,) * 2)
        eroded = cv2.erode(mask, k)
        if cv2.countNonZero(eroded) > 0.35 * cv2.countNonZero(mask):
            body = eroded
        else:
            warnings.append("erosion frange ignoree (masque trop fin)")

    contours, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("masque vide : aucun tapis detecte")
    cnt = max(contours, key=cv2.contourArea)

    # On cherche un quadrilatere en faisant varier la tolerance de simplification.
    peri = cv2.arcLength(cnt, True)
    quad = None
    for eps in np.linspace(0.005, 0.08, 24):
        approx = cv2.approxPolyDP(cnt, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2).astype(np.float32)
            break

    if quad is None:
        # Repli : rectangle oriente minimal. Moins precis si le tapis est deforme,
        # mais toujours exploitable. On le signale, le QC s'en servira.
        box = cv2.boxPoints(cv2.minAreaRect(cnt))
        quad = np.asarray(box, dtype=np.float32)
        warnings.append(
            "coins non detectes proprement : repli sur le rectangle minimal "
            "(tapis plie, occlus ou tres irregulier ?)"
        )
    return order_quad(quad), warnings


def _expand_quad(quad: np.ndarray, pad_ratio: float) -> np.ndarray:
    """Dilate le quad autour de son centre pour rattraper les franges."""
    c = quad.mean(axis=0)
    return (c + (quad - c) * (1.0 + pad_ratio)).astype(np.float32)


def measure_aspect_ratio(quad: np.ndarray) -> float:
    """Ratio apparent L/l mesure sur les bords. Utile pour recouper le DNA.

    Attention : c'est le ratio *apparent*. Sur une photo en perspective il est
    biaise. Il sert de garde-fou (detecter un DNA absurde), pas de verite.
    """
    tl, tr, br, bl = np.asarray(quad, dtype=np.float64)
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    long_, short = max(w, h), min(w, h)
    return float(long_ / max(short, 1e-6))


def extract_plate(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    aspect_ratio: float,
    target_long_px: int = 1600,
    fringe_pad_ratio: float = 0.04,
    fringe_erode_px: int = 6,
) -> Plate:
    """Photo + masque -> orthophoto RGBA du tapis.

    aspect_ratio vient du RUG DNA (ou des dimensions reelles declarees, qui sont
    toujours prioritaires). On ne "devine" pas le ratio depuis la photo : une
    photo en perspective ecrase le tapis, et redresser sur un ratio mesure a
    l'ecran reproduirait cette deformation dans le rendu final.
    """
    warnings: list[str] = []
    mask = _largest_component(mask)
    quad, w1 = _corners_from_mask(mask, fringe_erode_px)
    warnings += w1

    apparent = measure_aspect_ratio(quad)
    if apparent > 0 and not (0.45 < apparent / aspect_ratio < 2.2):
        warnings.append(
            f"ratio DNA ({aspect_ratio:.2f}) tres eloigne du ratio apparent "
            f"({apparent:.2f}) : verifier le RUG DNA ou l'angle de la photo"
        )

    padded = _expand_quad(quad, fringe_pad_ratio)

    if aspect_ratio >= 1.0:
        out_h, out_w = target_long_px, int(round(target_long_px / aspect_ratio))
    else:
        out_w, out_h = target_long_px, int(round(target_long_px * aspect_ratio))

    # Le tapis est-il photographie en portrait ou en paysage ? On aligne la
    # destination sur l'orientation source pour ne pas le faire pivoter de 90 deg.
    tl, tr, br, bl = padded
    src_w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    src_h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    if (src_w > src_h) != (out_w > out_h):
        out_w, out_h = out_h, out_w

    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], np.float32)
    H = cv2.getPerspectiveTransform(padded, dst)

    bgr = cv2.warpPerspective(
        image_bgr, H, (out_w, out_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    alpha = cv2.warpPerspective(mask, H, (out_w, out_h), flags=cv2.INTER_LINEAR)

    # Adoucir d'un cheveu le bord pour eviter l'escalier, sans "parfaire" la forme :
    # les irregularites reelles du tapis doivent survivre.
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

    rgba = np.dstack([bgr, alpha]).astype(np.uint8)
    pad_px = int(round(fringe_pad_ratio * max(out_w, out_h)))
    return Plate(rgba, quad, aspect_ratio, pad_px, warnings)


def pick_plate_photo(dna) -> str:
    """Choisit la photo source de la plate parmi les N vues.

    Critere : tapis entier, a plat, net, occupant le cadre. Une vue de detail ou
    un angle rasant donne une orthophoto molle et une geometrie fausse.
    """
    candidates = [p for p in dna.photos if p.usable_as_plate and p.shows_full_rug]
    if not candidates:
        candidates = [p for p in dna.photos if p.shows_full_rug] or list(dna.photos)
    best = max(
        candidates,
        key=lambda p: (
            (1.0 - p.perspective_severity) * 2.0 + p.sharpness * 1.5 + p.rug_coverage
        ),
    )
    return best.filename
