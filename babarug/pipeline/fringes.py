"""Recuperation des franges apres detourage.

Probleme constate sur BABA-RUG-0001 : GrabCut isole parfaitement le champ tisse
et coupe net les DEUX franges, parce qu'elles sont creme sur un mur creme. Le
controle de qualite de masque ne le voyait pas non plus : son test de contraste
est domine par les grands cotes (bordure rouge contre mur clair), et la perte
sur les petits cotes y est invisible.

Ce que la mesure a montre sur la photo reelle : les franges ne se distinguent
pas par la TEXTURE (variance locale 32, contre 20 a 40 pour la brique du mur :
indiscernable) mais par la COULEUR -- brins beige chaud contre mur rose-blanc
froid, ecart b* de 12,5 et ecart de luminance de 22,8 en LAB.

D'ou la methode : on modelise le FOND a partir de la zone exterieure de la
bande, ou il n'y a certainement pas de frange, puis on marque comme frange tout
ce qui s'en ecarte significativement ET qui touche le bord du tapis. Modeliser
le fond plutot que la frange rend la methode independante de la couleur du
tapis : elle marche avec des franges plus claires comme plus sombres que le mur.

Limite assumee : sur un fond de meme teinte ET de meme clarte que les franges,
l'information n'est pas dans l'image et rien ne la recuperera. La fonction le
signale au lieu d'inventer.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FringeResult:
    mask: np.ndarray            # masque complet : corps + franges
    fringe_only: np.ndarray     # les franges seules, pour inspection
    found: bool
    extent_px: tuple[int, int]  # longueur detectee (petit cote 1, petit cote 2)
    separation: float           # distance couleur frange/fond, en ecarts-types
    notes: list[str]


def _short_edge_bands(body: np.ndarray, extent: int):
    """Bandes exterieures adjacentes aux deux PETITS cotes du tapis.

    Les franges naissent des fils de chaine : elles ne peuvent apparaitre que
    sur les petits cotes. Chercher ailleurs ramasserait de l'ombre portee.
    """
    cnts, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    (cx, cy), (w, h), angle = rect
    if w < h:                      # normalise : w = GRAND cote, h = petit cote
        w, h, angle = h, w, angle + 90

    # Les petits cotes sont perpendiculaires au GRAND axe : pour deborder au-dela
    # d'eux, il faut allonger le rectangle selon w, pas selon h. Inverser les
    # deux fait couvrir tout le perimetre et ramasse les lisieres et l'ombre
    # portee des grands cotes -- erreur reellement commise ici.
    box = cv2.boxPoints(((cx, cy), (w + 2 * extent, h), angle))
    outer = np.zeros_like(body)
    cv2.fillPoly(outer, [box.astype(np.int32)], 255)
    band = cv2.bitwise_and(outer, cv2.bitwise_not(body))

    # Separe les deux extremites selon leur position le long du GRAND axe.
    ang = np.radians(angle)
    axis = np.array([np.cos(ang), np.sin(ang)])
    ys, xs = np.nonzero(band)
    if len(xs) == 0:
        return band, None, None
    proj = (xs - cx) * axis[0] + (ys - cy) * axis[1]
    side1 = np.zeros_like(band)
    side2 = np.zeros_like(band)
    side1[ys[proj < 0], xs[proj < 0]] = 255
    side2[ys[proj >= 0], xs[proj >= 0]] = 255
    return band, side1, side2


def recover_fringes(
    image_bgr: np.ndarray,
    body_mask: np.ndarray,
    max_extent_ratio: float = 0.055,
    min_separation: float = 2.0,
    work_px: int = 2200,
) -> FringeResult:
    """Etend le masque du corps du tapis pour y inclure ses franges.

    `work_px` borne le cote long utilise pour la classification. Sur une photo
    4000x6000 le calcul pleine resolution prenait ~27 s par tapis, ce qui est
    prohibitif en traitement de lot, pour un gain nul : un brin de frange fait
    encore plusieurs pixels de large a 2200 px. Le masque est ensuite remis a la
    resolution d'origine.
    """
    notes: list[str] = []
    full_shape = body_mask.shape[:2]
    orig_mask = body_mask
    vide = lambda: np.zeros(full_shape, np.uint8)
    scale = min(1.0, work_px / max(full_shape))
    if scale < 1.0:
        image_bgr = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        body_mask = cv2.resize(body_mask, (image_bgr.shape[1], image_bgr.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
    body = (body_mask > 127).astype(np.uint8) * 255
    ys, xs = np.nonzero(body)
    if len(xs) == 0:
        return FringeResult(orig_mask, vide(), False, (0, 0), 0.0,
                            ["masque de corps vide"])

    base = int(max(ys.max() - ys.min(), xs.max() - xs.min()) * max_extent_ratio)
    base = max(12, base)

    # L'etendue de frange n'est pas connue a l'avance. On part d'une valeur
    # proportionnelle au tapis, puis on l'elargit tant que l'anneau servant de
    # modele de FOND reste heterogene : tant que des brins y trainent, sa
    # dispersion colorimetrique reste elevee. Une frange inhabituellement longue
    # n'invalide donc plus la mesure, elle deplace seulement la zone de reference.
    lab_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    body_dist = cv2.distanceTransform((body == 0).astype(np.uint8), cv2.DIST_L2, 5)
    # L'anneau de reference est echantillonne AU-DELA de la bande de recherche
    # (entre 1.0 et 1.8 fois son etendue), jamais dedans : c'est ce qui rend la
    # contamination par les brins structurellement impossible.
    #
    # Une version adaptative a ete essayee, qui eloignait l'anneau tant que le
    # fond n'etait pas homogene. Elle reglait un cas synthetique de franges
    # anormalement longues, mais degradait la vraie photo : la dispersion du
    # fond y chutant, les distances s'envolaient, l'algorithme happait l'ombre
    # portee sous le tapis (profondeur mesuree 302 px au lieu de 180) et la
    # silhouette redevenait rectiligne. Elle a donc ete retiree : un tapis dont
    # les franges depassent l'etendue nominale est signale, pas devine.
    extent = base
    ring_extent = base

    band, side1, side2 = _short_edge_bands(body, extent)
    if side1 is None:
        return FringeResult(orig_mask, vide(), False, (0, 0), 0.0,
                            ["bande de recherche vide"])

    # On ne convertit et ne classe que la boite englobante de la bande : sur une
    # photo 4000x6000, traiter l'image entiere coutait ~27 s par tapis.
    bys, bxs = np.nonzero(cv2.bitwise_or(
        band, _short_edge_bands(body, int(ring_extent * 1.8))[0]))
    ry0, ry1 = max(0, bys.min() - 2), min(band.shape[0], bys.max() + 3)
    rx0, rx1 = max(0, bxs.min() - 2), min(band.shape[1], bxs.max() + 3)
    roi = (slice(ry0, ry1), slice(rx0, rx1))
    lab = np.zeros((*band.shape, 3), np.float32)
    lab[roi] = lab_full[roi]

    # Distance de chaque pixel au corps du tapis : c'est elle qui definit
    # proprement "pres du tapis" (frange probable) et "loin" (fond certain).
    outside = body_dist
    in_band = band > 0

    # Modele de FOND : echantillonne AU-DELA de la bande de recherche, entre
    # 1.0 et 1.8 fois l'extension maximale. Prelever le fond a l'interieur de la
    # bande revient a le prendre au milieu des brins quand la frange est longue :
    # le modele adopte alors la couleur de la frange et l'ecart mesure s'effondre.
    # En echantillonnant plus loin que ce qu'on accepte de marquer comme frange,
    # la contamination devient structurellement impossible.
    outer, _, _ = _short_edge_bands(body, int(ring_extent * 1.8))
    far = (outer > 0) & (outside > ring_extent * 1.02)
    if far.sum() < 200:
        far = (outer > 0) & (outside > ring_extent * 0.75)
    if far.sum() < 200:
        return FringeResult(orig_mask, vide(), False, (0, 0), 0.0,
                            ["pas assez de fond de reference pour modeliser l'arriere-plan"])

    mu = lab[far].mean(axis=0)
    sigma = np.maximum(lab[far].std(axis=0), 1.5)

    # Distance au fond, en ecarts-types. La luminance et l'axe jaune-bleu portent
    # l'essentiel du signal sur une frange de laine ecrue.
    d = np.abs(lab - mu) / sigma
    dist = np.sqrt((d[..., 0] ** 2) * 1.0 + (d[..., 1] ** 2) * 0.7 + (d[..., 2] ** 2) * 1.3)

    # Zone de GRAINE : tiers interieur, contre le bord du tapis, ou tous les
    # brins sont presents. On y mesure la separation par un percentile haut et
    # non par la mediane : des brins fins laissent voir le fond entre eux, et
    # une mediane mesurerait alors le fond au lieu de la frange.
    seed_zone = in_band & (outside < 0.35 * extent)
    if seed_zone.sum() < 100:
        seed_zone = in_band
    separation = float(np.percentile(dist[seed_zone], 75))

    if separation < min_separation:
        notes.append(
            f"franges indiscernables du fond (separation {separation:.1f} ecart-type) : "
            "le tapis et l'arriere-plan ont la meme couleur et la meme clarte. "
            "Rephotographier sur un fond contrastant -- l'information n'est pas "
            "dans cette image."
        )
        return FringeResult(orig_mask, vide(), False, (0, 0), separation, notes)

    seuil = max(min_separation * 0.8, separation * 0.45)
    fringe = ((dist > seuil) & in_band).astype(np.uint8) * 255

    # Une frange part du tapis : on ne garde que ce qui lui est rattache.
    seeded = cv2.bitwise_or(fringe, cv2.dilate(
        body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats((seeded > 0).astype(np.uint8), 8)
    if n > 1:
        keep = np.unique(lbl[body > 0])
        keep = keep[keep != 0]
        seeded = np.isin(lbl, keep).astype(np.uint8) * 255
    fringe = cv2.bitwise_and(seeded, band)

    # Nettoyage LEGER : on retire les mouchetures isolees, on ne lisse pas les
    # bouts de brins -- leur irregularite fait partie du produit.
    fringe = cv2.morphologyEx(fringe, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # Ne garder que ce qui touche vraiment le tapis : une frange part du bord.
    near = cv2.dilate(body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats((fringe > 0).astype(np.uint8), 8)
    if n > 1:
        touching = np.unique(lbl[(near > 0) & (fringe > 0)])
        touching = touching[touching != 0]
        big = {i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= extent * 4}
        keep = [i for i in range(1, n) if i in set(touching.tolist()) and i in big]
        fringe = np.isin(lbl, keep).astype(np.uint8) * 255

    # Boucher les TROUS INTERIEURS sans toucher a la silhouette exterieure.
    # Les brins les plus clairs passent pour du mur et laissent des lacunes au
    # milieu de la frange. On remplit uniquement ce qui est entierement encercle
    # par de la frange : le contour exterieur, lui, garde ses dents.
    holes = cv2.bitwise_not(fringe)
    ff = holes.copy()
    hh, ww = ff.shape
    flood = np.zeros((hh + 2, ww + 2), np.uint8)
    cv2.floodFill(ff, flood, (0, 0), 0)
    for seed in ((ww - 1, 0), (0, hh - 1), (ww - 1, hh - 1)):
        cv2.floodFill(ff, flood, seed, 0)
    fringe = cv2.bitwise_or(fringe, cv2.bitwise_and(ff, band))

    # Profondeur de frange = distance maximale au corps du tapis, mesuree par
    # transformee de distance. Plus fiable qu'une boite englobante, qui melange
    # la longueur des brins et la largeur du tapis.
    outside = cv2.distanceTransform((body == 0).astype(np.uint8), cv2.DIST_L2, 5)

    def _depth(side):
        z = (cv2.bitwise_and(fringe, side) > 0)
        if z.sum() < 50:
            return 0
        return int(np.percentile(outside[z], 97))

    e1, e2 = _depth(side1), _depth(side2)
    found = (fringe > 0).sum() > extent * 20
    if found and (e1 == 0 or e2 == 0):
        notes.append("frange retrouvee sur un seul petit cote : verifier l'autre extremite")
    if not found:
        notes.append("aucune frange retrouvee malgre un contraste suffisant")

    full = cv2.bitwise_or(body, fringe)
    if scale < 1.0:
        inv = 1.0 / scale
        up = lambda m: cv2.resize(m, (full_shape[1], full_shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        full, fringe = up(full), up(fringe)
        e1, e2 = int(e1 * inv), int(e2 * inv)
    return FringeResult(full, fringe, found, (e1, e2), separation, notes)
