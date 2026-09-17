"""Etape 6 : le CONTROLE QUALITE.

Deux niveaux, volontairement separes :

1. FIDELITE PRODUIT -- mesuree, deterministe, sans IA. On re-extrait le tapis de
   l'image finale (homographie inverse) et on le compare a la plate source :
   ecart colorimetrique CIEDE2000, SSIM, appariement de points d'interet,
   histogramme, occultation, geometrie. Ces chiffres sont reproductibles et
   opposables. C'est le seul juge qui peut REJETER.

2. REALISME DE SCENE -- avis d'un modele de vision. Consultatif. Il peut
   declencher une correction ou faire passer en REVIEW_REQUIRED, il ne peut
   JAMAIS repecher une image qui a echoue au niveau 1.

Ce decoupage repond directement a "ne considere jamais un score d'IA comme une
preuve absolue".
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.color import rgb2lab
from skimage.metrics import structural_similarity

from babarug.geometry import order_quad
from babarug.models import FidelityMetrics, QCReport, SceneReview, Verdict
from babarug.pipeline.composite import recover_plate
from babarug.pipeline.plate import Plate


@dataclass
class QCThresholds:
    """Seuils de production. Calibrer sur les 5-10 premiers tapis reels avant
    de les figer : les valeurs ci-dessous sont un point de depart raisonnable,
    pas une verite universelle."""

    delta_e_mean_reject: float = 6.0
    delta_e_mean_review: float = 3.0
    delta_e_p95_reject: float = 12.0
    ssim_reject: float = 0.80
    ssim_review: float = 0.92
    keypoint_reject: float = 0.45
    keypoint_review: float = 0.65
    histogram_review: float = 0.95
    visible_reject: float = 0.90
    visible_review: float = 0.985
    scale_error_reject: float = 0.12
    scale_error_review: float = 0.06
    texture_reject: float = 0.55
    texture_review: float = 0.78
    silhouette_reject: float = 0.90
    silhouette_review: float = 0.975
    min_keypoints: int = 30


# ------------------------------------------------------------------ mesures

def _delta_e(a_bgr: np.ndarray, b_bgr: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """CIEDE2000 moyen et p95 sur la zone du tapis.

    CIEDE2000 et pas une distance RGB : c'est la metrique qui correspond a ce
    que l'oeil percoit. dE < 1 imperceptible, < 2-3 invisible en pratique,
    > 5 visible cote a cote.
    """
    m = mask > 127
    if not m.any():
        return 0.0, 0.0
    a = rgb2lab(cv2.cvtColor(a_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    b = rgb2lab(cv2.cvtColor(b_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    from skimage.color import deltaE_ciede2000

    d = deltaE_ciede2000(a, b)[m]
    return float(np.mean(d)), float(np.percentile(d, 95))


def _ssim(a_bgr: np.ndarray, b_bgr: np.ndarray, mask: np.ndarray) -> float:
    ga = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2GRAY)
    _, smap = structural_similarity(ga, gb, full=True)
    m = mask > 127
    return float(smap[m].mean()) if m.any() else 1.0


def _keypoint_ratio(a_bgr: np.ndarray, b_bgr: np.ndarray, mask: np.ndarray, min_kp: int):
    """Fidelite du MOTIF : les points d'interet du tapis d'origine se
    retrouvent-ils, au bon endroit, dans le rendu ?

    C'est la mesure qui attrape le pire scenario du projet : une image superbe
    ou le generateur a redessine un medaillon central different. Les couleurs
    et l'histogramme peuvent rester corrects ; l'appariement, non.
    """
    ga = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2GRAY)
    m = (mask > 127).astype(np.uint8) * 255

    orb = cv2.ORB_create(nfeatures=3000)
    ka, da = orb.detectAndCompute(ga, m)
    kb, db = orb.detectAndCompute(gb, m)
    if da is None or db is None or len(ka) < min_kp or len(kb) < min_kp:
        return 1.0, False  # non mesurable -> ne pas faire croire a un succes

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = matcher.knnMatch(da, db, k=2)
    good = [p[0] for p in raw if len(p) == 2 and p[0].distance < 0.78 * p[1].distance]
    if len(good) < min_kp:
        return 0.0, True

    src = np.float32([ka[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
    _, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if inliers is None:
        return 0.0, True
    return float(inliers.sum()) / float(len(good)), True


def _texture_retention(a_bgr: np.ndarray, b_bgr: np.ndarray, mask: np.ndarray) -> float:
    """Energie haute frequence conservee entre le tapis reel et le rendu.

    Cible un echec que la couleur et la structure ne voient pas : un tapis
    "nettoye". Un lissage, un debruitage agressif ou un modele generatif qui
    refait la laine effacent l'usure, l'abrash et le grain de noeud -- l'image
    reste belle, le produit n'est plus le bon. Le ratio d'ecart-type laplacien
    le chiffre directement.
    """
    m = mask > 127
    if m.sum() < 64:
        return 1.0
    la = cv2.Laplacian(cv2.cvtColor(a_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_32F)
    lb = cv2.Laplacian(cv2.cvtColor(b_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_32F)
    ea, eb = float(la[m].std()), float(lb[m].std())
    if ea < 1e-3:
        return 1.0
    return float(np.clip(eb / ea, 0.0, 2.0))


def _silhouette_iou(plate: Plate, H: np.ndarray, rendered_alpha, img_shape) -> float:
    """Concordance entre la silhouette attendue du tapis et celle obtenue.

    Garde-fou contre le detourage "trop parfait" : franges rabotees, coins
    arrondis, bord replie redresse. Le cahier des charges demande explicitement
    de conserver ces irregularites -- il faut donc pouvoir les mesurer.
    """
    if rendered_alpha is None:
        return 1.0
    h, w = img_shape[:2]
    expected = cv2.warpPerspective(plate.rgba[:, :, 3], H, (w, h), flags=cv2.INTER_NEAREST) > 127
    got = np.asarray(rendered_alpha)
    if got.shape[:2] != (h, w):
        got = cv2.resize(got, (w, h), interpolation=cv2.INTER_NEAREST)
    got = got > 127
    union = (expected | got).sum()
    return float((expected & got).sum() / union) if union else 1.0


def _histogram_corr(a_bgr: np.ndarray, b_bgr: np.ndarray, mask: np.ndarray) -> float:
    m = (mask > 127).astype(np.uint8)
    if not m.any():
        return 1.0
    scores = []
    for ch in range(3):
        ha = cv2.calcHist([a_bgr], [ch], m, [64], [0, 256])
        hb = cv2.calcHist([b_bgr], [ch], m, [64], [0, 256])
        cv2.normalize(ha, ha)
        cv2.normalize(hb, hb)
        scores.append(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))
    return float(np.mean(scores))


def _visible_fraction(plate: Plate, H: np.ndarray, img_shape) -> float:
    """Part du tapis reellement presente dans le cadre final.

    On projette l'emprise du tapis sur un canevas elargi qui contient tout le
    quad, meme debordant, puis on regarde quelle fraction tombe dans l'image.
    Compter les pixels de la plate et ceux de la scene reviendrait a mesurer le
    facteur de reduction, pas l'occultation -- erreur facile et silencieuse.
    """
    h, w = img_shape[:2]
    ph, pw = plate.rgba.shape[:2]
    alpha = plate.rgba[:, :, 3]

    corners = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]]).reshape(1, 4, 2)
    proj = cv2.perspectiveTransform(corners, H).reshape(4, 2)
    x0, y0 = np.floor(np.minimum(proj.min(axis=0), [0, 0])).astype(int)
    x1, y1 = np.ceil(np.maximum(proj.max(axis=0), [w, h])).astype(int)
    cw, ch = int(x1 - x0), int(y1 - y0)
    if cw <= 0 or ch <= 0 or cw * ch > 80_000_000:
        return 0.0

    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64)
    full = cv2.warpPerspective(alpha, T @ H, (cw, ch), flags=cv2.INTER_NEAREST)
    total = float((full > 127).sum())
    if total == 0:
        return 0.0
    inside = full[-y0 : -y0 + h, -x0 : -x0 + w] if (-y0 >= 0 and -x0 >= 0) else full
    return float(min(1.0, (inside > 127).sum() / total))


def geometry_error(rug_quad, floor, aspect_ratio: float) -> float:
    """Ecart relatif entre le ratio restitue au sol et le ratio du RUG DNA.

    Mesure faite dans le plan du sol remis a l'echelle reelle : la perspective
    y est annulee, donc un ecart ici est une vraie deformation du produit.
    """
    from babarug.geometry import FloorPlane, rug_quad_metrics

    if not isinstance(floor, FloorPlane):
        floor = FloorPlane(np.asarray(floor, dtype=np.float32))
    L, W = rug_quad_metrics(rug_quad, floor)
    got = L / max(W, 1e-6)
    return float(abs(got - aspect_ratio) / max(aspect_ratio, 1e-6))


def _match_scale(plate_rgba: np.ndarray, recovered: np.ndarray, rug_quad) -> tuple:
    """Ramene la comparaison a la resolution que le tapis occupe VRAIMENT.

    La plate fait 1600 px de long ; dans un 1600x900 le tapis n'en occupe
    peut-etre que 600. Comparer a pleine resolution reviendrait a sanctionner
    une reduction d'echelle parfaitement normale comme une perte de motif. On
    compare donc au niveau de detail reellement livre au client.
    """
    from babarug.geometry import quad_area

    ph, pw = plate_rgba.shape[:2]
    if rug_quad is None:
        return plate_rgba[:, :, :3], recovered, plate_rgba[:, :, 3], 1.0
    scale = float(np.sqrt(quad_area(rug_quad) / max(pw * ph, 1)))
    scale = float(np.clip(scale, 0.08, 1.0))
    if scale > 0.97:
        return plate_rgba[:, :, :3], recovered, plate_rgba[:, :, 3], 1.0
    nw, nh = max(32, int(pw * scale)), max(32, int(ph * scale))
    rs = lambda im, interp=cv2.INTER_AREA: cv2.resize(im, (nw, nh), interpolation=interp)
    return rs(plate_rgba[:, :, :3]), rs(recovered), rs(plate_rgba[:, :, 3]), scale


def _normalize_shading(original: np.ndarray, recovered: np.ndarray, mask: np.ndarray):
    """Annule l'eclairage de la piece avant de comparer les couleurs.

    Le relighting est VOULU : un tapis doit recevoir la lumiere de la scene. Sans
    le neutraliser, le dE mesure l'eclairage et non une derive produit.

    Deux precautions, toutes deux apprises a leurs depens :
      - le flou est MASQUE (on divise la somme ponderee par le poids) : un flou
        ordinaire fait deborder le sol environnant dans le champ d'eclairage du
        tapis et injecte une erreur qui n'existe pas ;
      - le rayon est proportionnel a la taille du TAPIS, pas de l'image. Un rayon
        calcule sur une scene en 1600x900 valait 163 px pour un tapis large de
        260 px : le champ estime n'avait plus rien a voir avec lui, et le dE
        mesure restait a 5.5 sur un compositing pourtant strictement identique
        a sa reference.
    """
    m = (mask > 127).astype(np.float32)
    if m.sum() < 64:
        return recovered, 0.0
    lo = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32) + 1.0
    lr = cv2.cvtColor(recovered, cv2.COLOR_BGR2GRAY).astype(np.float32) + 1.0

    r = max(9, (int(np.sqrt(float(m.sum())) * 0.25) // 2) * 2 + 1)
    w = cv2.GaussianBlur(m, (r, r), 0)
    num = cv2.GaussianBlur(lr * m, (r, r), 0)
    den = cv2.GaussianBlur(lo * m, (r, r), 0)
    ok = w > 1e-3
    field = np.ones_like(lo)
    field[ok] = np.clip(num[ok] / np.maximum(den[ok], 1e-3), 0.4, 2.5)

    strength = float(np.abs(field[m > 0.5] - 1.0).mean())
    corrected = np.clip(recovered.astype(np.float32) / field[..., None], 0, 255).astype(np.uint8)
    return corrected, strength


# ------------------------------------------------------------------ verdict

def measure_fidelity(
    plate: Plate,
    final_bgr: np.ndarray,
    H_plate_to_scene: np.ndarray,
    rug_quad=None,
    floor=None,
    rendered_alpha=None,
    thresholds: QCThresholds | None = None,
) -> tuple[FidelityMetrics, list[str]]:
    """Compare le tapis rendu au tapis reel. Aucun appel modele.

    La comparaison se fait EN ESPACE-SCENE : on projette la plate d'origine dans
    le quad du sol -- exactement la transformation qu'applique le compositing --
    et on compare cette reference au rendu final, la ou le tapis se trouve.

    Une premiere version ramenait au contraire le rendu dans l'espace de la
    plate par homographie inverse. C'etait biaise : le tapis n'occupant que ~16 %
    de l'echelle de la plate dans une vue large, le rendu subissait un
    reechantillonnage aller-retour que l'original ne subissait pas. On mesurait
    alors une perte de resolution parfaitement normale comme une infidelite, et
    des images correctes etaient rejetees. En espace-scene les deux images ont la
    meme histoire de reechantillonnage : l'ecart qui subsiste est reel.
    """
    th = thresholds or QCThresholds()
    notes: list[str] = []
    h, w = final_bgr.shape[:2]

    expected = cv2.warpPerspective(plate.rgba, H_plate_to_scene, (w, h),
                                   flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0, 0))
    original = expected[:, :, :3]
    alpha = expected[:, :, 3]

    # On ecarte une bande au bord : feather, ombre et liseré de contact y
    # melangent legitimement tapis et sol.
    ksz = max(3, (int(np.sqrt((alpha > 127).sum()) * 0.02) // 2) * 2 + 1)
    core = cv2.erode(alpha, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz)))
    if (core > 127).sum() < 256:
        core = alpha
    if (core > 127).sum() < 64:
        notes.append("tapis quasi absent du rendu : controle impossible")
        return FidelityMetrics(delta_e_mean=99, delta_e_p95=99, ssim=0, keypoint_inlier_ratio=0,
                               histogram_correlation=0, visible_fraction=0, scale_error=0,
                               texture_retention=0, silhouette_iou=0), notes

    recovered_flat, shading = _normalize_shading(original, final_bgr, core)
    if shading > 0.28:
        notes.append(f"relighting tres marque ({shading:.0%}) : verifier que le tapis "
                     "n'est pas delave ou assombri par la scene")

    de_mean, de_p95 = _delta_e(original, recovered_flat, core)
    ssim = _ssim(original, recovered_flat, core)
    kp, measurable = _keypoint_ratio(original, recovered_flat, core, th.min_keypoints)
    if not measurable:
        notes.append(
            "fidelite du motif non mesurable (tapis trop uni, ou rendu trop petit) "
            "-> validation humaine requise"
        )
    hist = _histogram_corr(original, recovered_flat, core)
    texture = _texture_retention(original, recovered_flat, core)
    silhouette = _silhouette_iou(plate, H_plate_to_scene, rendered_alpha, final_bgr.shape)
    visible = _visible_fraction(plate, H_plate_to_scene, final_bgr.shape)
    scale_err = (
        geometry_error(rug_quad, floor, plate.aspect_ratio)
        if rug_quad is not None and floor is not None
        else 0.0
    )

    # Lisibilite du produit : independante de la fidelite, mais decisive pour une
    # fiche e-commerce. Un tapis fidele mais minuscule ne montre pas le produit.
    if rug_quad is not None:
        from babarug.geometry import quad_area

        occupe = quad_area(rug_quad) / float(w * h)
        if occupe < 0.10:
            notes.append(
                f"le tapis n'occupe que {occupe:.1%} de l'image : cadrage trop large, "
                "le motif sera peu lisible sur la fiche produit"
            )

    return (
        FidelityMetrics(
            delta_e_mean=de_mean,
            delta_e_p95=de_p95,
            ssim=ssim,
            keypoint_inlier_ratio=kp,
            histogram_correlation=hist,
            visible_fraction=visible,
            scale_error=scale_err,
            texture_retention=texture,
            silhouette_iou=silhouette,
        ),
        notes,
    )


def decide(
    variant: str,
    metrics: FidelityMetrics,
    review: SceneReview | None = None,
    extra_notes: list[str] | None = None,
    thresholds: QCThresholds | None = None,
    iteration: int = 0,
) -> QCReport:
    """Combine mesures et avis en un verdict. La fidelite prime, toujours."""
    th = thresholds or QCThresholds()
    failures: list[str] = list(extra_notes or [])
    reject = False
    review_needed = bool(failures)

    def _fail(cond_reject: bool, cond_review: bool, msg: str):
        nonlocal reject, review_needed
        if cond_reject:
            failures.append(f"[REJET] {msg}")
            reject = True
        elif cond_review:
            failures.append(f"[A VERIFIER] {msg}")
            review_needed = True

    _fail(
        metrics.delta_e_mean > th.delta_e_mean_reject,
        metrics.delta_e_mean > th.delta_e_mean_review,
        f"derive colorimetrique dE={metrics.delta_e_mean:.2f}",
    )
    _fail(metrics.delta_e_p95 > th.delta_e_p95_reject, False,
          f"derive colorimetrique locale dE p95={metrics.delta_e_p95:.2f}")
    _fail(
        metrics.ssim < th.ssim_reject,
        metrics.ssim < th.ssim_review,
        f"structure du tapis alteree SSIM={metrics.ssim:.3f}",
    )
    _fail(
        metrics.keypoint_inlier_ratio < th.keypoint_reject,
        metrics.keypoint_inlier_ratio < th.keypoint_review,
        f"motif non retrouve (appariement={metrics.keypoint_inlier_ratio:.2f})",
    )
    _fail(False, metrics.histogram_correlation < th.histogram_review,
          f"distribution des couleurs decalee (corr={metrics.histogram_correlation:.3f})")
    _fail(
        metrics.visible_fraction < th.visible_reject,
        metrics.visible_fraction < th.visible_review,
        f"tapis partiellement hors cadre ({metrics.visible_fraction:.1%} visible)",
    )
    _fail(
        metrics.texture_retention < th.texture_reject,
        metrics.texture_retention < th.texture_review,
        f"texture/usure effacee (energie de detail conservee : "
        f"{metrics.texture_retention:.0%}) -- le tapis parait artificiellement neuf",
    )
    _fail(
        metrics.silhouette_iou < th.silhouette_reject,
        metrics.silhouette_iou < th.silhouette_review,
        f"silhouette alteree (IoU={metrics.silhouette_iou:.3f}) : franges ou bords "
        "irreguliers rabotes",
    )
    _fail(
        metrics.scale_error > th.scale_error_reject,
        metrics.scale_error > th.scale_error_review,
        f"proportions du tapis deviees de {metrics.scale_error:.1%}",
    )

    # Avis de scene : peut degrader, jamais ameliorer.
    if review is not None:
        if review.rug_merges_with_furniture:
            failures.append("[REJET] le tapis fusionne avec le mobilier")
            reject = True
        if not review.rug_fully_visible:
            failures.append("[A VERIFIER] le tapis n'est pas entierement visible")
            review_needed = True
        for label, score in (
            ("realisme de scene", review.scene_realism),
            ("perspective", review.perspective),
            ("integration du tapis", review.integration),
        ):
            if score < 50:
                failures.append(f"[A VERIFIER] {label} faible ({score}/100)")
                review_needed = True
        for p in review.problems:
            failures.append(f"[A VERIFIER] {p}")
            review_needed = True

    verdict = Verdict.REJECTED if reject else (
        Verdict.REVIEW_REQUIRED if review_needed else Verdict.APPROVED
    )
    return QCReport(
        variant=variant, verdict=verdict, fidelity=metrics,
        review=review, failures=failures, iteration=iteration,
    )
