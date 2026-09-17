"""Etape 5 : le COMPOSITING.

Principe directeur : les pixels du tapis dans l'image finale sont les pixels de
la plate, transformes par une homographie. Aucun modele generatif ne redessine
le tapis. La fidelite n'est donc pas une esperance statistique, c'est une
propriete de construction.

Le travail ici consiste a rendre cette insertion CREDIBLE :
  - relighting  : le tapis recoit l'eclairage de la piece
  - ombre portee: contact au sol
  - grain/flou  : coherence optique avec la photo de scene

Regle de couleur : par defaut on ne module que la LUMINANCE. Aucune derive de
teinte n'est appliquee au tapis, parce qu'une image jolie qui a rechauffe les
rouges d'un Kazak est un echec produit. `chroma_adapt` existe pour les cas ou
l'on accepte explicitement ce compromis ; il vaut 0 par defaut.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from babarug.geometry import order_quad
from babarug.pipeline.plate import Plate


@dataclass
class CompositeOptions:
    relight_strength: float = 0.85   # 0 = tapis brut, 1 = suit totalement la lumiere du sol
    chroma_adapt: float = 0.0        # derive de teinte. NE PAS monter sans raison.
    shadow_opacity: float = 0.42
    shadow_blur_px: int = 25
    shadow_offset: tuple[int, int] = (0, 6)
    contact_darkening: float = 0.25  # assombrissement du liseré au contact du sol
    match_grain: bool = True
    match_blur: bool = True
    edge_feather_px: float = 0.8


@dataclass
class CompositeResult:
    image: np.ndarray            # (H,W,3) uint8 scene finale
    H_plate_to_scene: np.ndarray # homographie plate -> scene
    rug_alpha: np.ndarray        # (H,W) uint8, emprise du tapis dans la scene
    rug_quad: np.ndarray         # (4,2) quad effectivement utilise


# ---------------------------------------------------------------- outils bas niveau

def _luma(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _estimate_noise_sigma(img: np.ndarray) -> float:
    """Ecart-type du bruit haute frequence, via un laplacien (Immerkaer)."""
    g = _luma(img)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    conv = cv2.filter2D(g, -1, k)
    return float(np.abs(conv).mean() * np.sqrt(0.5 * np.pi) / 6.0)


def _estimate_blur(img: np.ndarray) -> float:
    """Variance du laplacien : eleve = net, bas = flou."""
    return float(cv2.Laplacian(_luma(img), cv2.CV_32F).var())


def _shading_field(scene: np.ndarray, rug_alpha: np.ndarray) -> np.ndarray:
    """Champ d'eclairage du sol a l'emplacement du tapis.

    Le sol sous le tapis est visible AVANT compositing (la scene est generee
    vide). On lisse fortement sa luminance pour ne garder que le gradient
    lumineux -- pas la texture du parquet, qui ne doit pas s'imprimer sur le
    tapis. Le champ est normalise a 1 en moyenne : il module, il ne reexpose pas.
    """
    lum = _luma(scene)
    # Rayon proportionnel a l'image : on veut le degrade, pas les lattes.
    r = max(31, (int(min(scene.shape[:2]) * 0.06) // 2) * 2 + 1)
    smooth = cv2.GaussianBlur(lum, (r, r), 0)
    m = rug_alpha > 8
    if not m.any():
        return np.ones(lum.shape, np.float32)
    ref = float(np.median(smooth[m]))
    if ref < 1e-3:
        return np.ones(lum.shape, np.float32)
    field = smooth / ref
    return np.clip(field, 0.55, 1.65).astype(np.float32)


def _contact_shadow(rug_alpha: np.ndarray, opt: CompositeOptions) -> np.ndarray:
    """Ombre douce + liseré de contact. Sans ca le tapis flotte."""
    a = rug_alpha.astype(np.float32) / 255.0
    dx, dy = opt.shadow_offset
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(a, M, (a.shape[1], a.shape[0]))
    k = max(3, (opt.shadow_blur_px // 2) * 2 + 1)
    soft = cv2.GaussianBlur(shifted, (k, k), 0) * opt.shadow_opacity

    # Liseré : la ou l'ombre est presente mais le tapis absent -> contact au sol.
    rim = cv2.GaussianBlur(a, (5, 5), 0) - a
    soft += np.clip(rim, 0, 1) * opt.contact_darkening
    return np.clip(soft, 0, 1)


# ---------------------------------------------------------------- compositing

def composite_rug(
    scene_bgr: np.ndarray,
    plate: Plate,
    rug_quad: np.ndarray,
    opt: CompositeOptions | None = None,
) -> CompositeResult:
    """Projette la plate dans le quad de sol et l'integre a la scene."""
    opt = opt or CompositeOptions()
    H_img, W_img = scene_bgr.shape[:2]
    rug_quad = order_quad(np.asarray(rug_quad, dtype=np.float32))

    ph, pw = plate.rgba.shape[:2]
    src = np.array([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]], np.float32)
    H = cv2.getPerspectiveTransform(src, rug_quad)

    warped = cv2.warpPerspective(
        plate.rgba, H, (W_img, H_img), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0, 0)
    )
    rug_bgr = warped[:, :, :3].astype(np.float32)
    rug_a = warped[:, :, 3].astype(np.float32) / 255.0

    if opt.edge_feather_px > 0:
        rug_a = cv2.GaussianBlur(rug_a, (0, 0), opt.edge_feather_px)

    scene = scene_bgr.astype(np.float32)

    # 1. Ombre portee : appliquee au SOL, avant de poser le tapis.
    shadow = _contact_shadow((rug_a * 255).astype(np.uint8), opt)
    scene = scene * (1.0 - shadow[..., None])

    # 2. Relighting : luminance seulement.
    field = _shading_field(scene_bgr, (rug_a * 255).astype(np.uint8))
    field = 1.0 + (field - 1.0) * opt.relight_strength
    rug_lit = rug_bgr * field[..., None]

    # 3. Adaptation chromatique optionnelle, volontairement desactivee par defaut.
    if opt.chroma_adapt > 0:
        m = rug_a > 0.5
        if m.any():
            floor_mean = scene_bgr.astype(np.float32)[m].mean(axis=0)
            rug_mean = rug_bgr[m].mean(axis=0)
            gain = np.clip(floor_mean / np.maximum(rug_mean, 1e-3), 0.85, 1.18)
            gain = gain / gain.mean()  # ne change pas l'exposition, juste la balance
            rug_lit = rug_lit * (1.0 + (gain - 1.0) * opt.chroma_adapt)

    # 4. Coherence optique.
    if opt.match_blur:
        scene_sharp = _estimate_blur(scene_bgr)
        rug_sharp = _estimate_blur(plate.rgba[:, :, :3])
        if rug_sharp > scene_sharp * 1.6 and scene_sharp > 0:
            sigma = float(np.clip(np.sqrt(rug_sharp / max(scene_sharp, 1e-6)) * 0.25, 0.3, 1.6))
            rug_lit = cv2.GaussianBlur(rug_lit, (0, 0), sigma)
    if opt.match_grain:
        sigma = _estimate_noise_sigma(scene_bgr)
        if sigma > 0.4:
            rng = np.random.default_rng(1234)  # deterministe : le QC doit etre reproductible
            rug_lit = rug_lit + rng.normal(0, sigma, rug_lit.shape).astype(np.float32)

    out = scene * (1.0 - rug_a[..., None]) + np.clip(rug_lit, 0, 255) * rug_a[..., None]
    out = np.clip(out, 0, 255).astype(np.uint8)
    return CompositeResult(out, H, (rug_a * 255).astype(np.uint8), rug_quad)


def recover_plate(
    final_bgr: np.ndarray, H_plate_to_scene: np.ndarray, plate_size: tuple[int, int]
) -> np.ndarray:
    """Re-extrait le tapis de l'image finale dans l'espace de la plate.

    C'est la cle du controle qualite : on remet le rendu a plat, puis on le
    compare pixel a pixel a la plate d'origine. Ca transforme "le tapis a-t-il
    change ?" en une mesure, pas en une opinion de modele.
    """
    pw, ph = plate_size
    return cv2.warpPerspective(
        final_bgr, np.linalg.inv(H_plate_to_scene), (pw, ph), flags=cv2.INTER_CUBIC
    )
