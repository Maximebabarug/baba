"""Generateurs d'images synthetiques : permettent de tester tout le pipeline
geometrique sans appeler une seule API et sans photo reelle."""

from __future__ import annotations

import cv2
import numpy as np


def synth_rug(w: int = 900, h: int = 1400, seed: int = 7) -> np.ndarray:
    """Tapis type oriental : champ, medaillon central, bordure, franges, abrash."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (38, 42, 130)  # BGR, rouge brique fonce

    # abrash (variation de bain de teinture) : irregularite reelle a preserver
    bands = cv2.resize(rng.normal(0, 9, (18, 3)).astype(np.float32), (w, h),
                       interpolation=cv2.INTER_CUBIC)
    img = np.clip(img.astype(np.float32) + bands[..., None][:, :, 0:1], 0, 255).astype(np.uint8)

    # bordures concentriques
    for i, col in enumerate([(30, 95, 160), (180, 180, 200), (40, 40, 90)]):
        t = 26 + i * 26
        cv2.rectangle(img, (t, t), (w - t, h - t), col, 16)

    # champ : semis de motifs boteh
    for gy in range(6):
        for gx in range(4):
            cx = int((gx + 0.5) * w / 4)
            cy = int((gy + 0.5) * h / 6)
            cv2.ellipse(img, (cx, cy), (26, 40), 12, 0, 360, (200, 205, 215), -1)
            cv2.ellipse(img, (cx, cy), (16, 26), 12, 0, 360, (40, 110, 175), -1)

    # medaillon central
    cv2.ellipse(img, (w // 2, h // 2), (150, 260), 0, 0, 360, (35, 70, 40), -1)
    cv2.ellipse(img, (w // 2, h // 2), (110, 210), 0, 0, 360, (190, 195, 210), 10)
    for k in range(8):
        a = k * np.pi / 4
        p = (int(w // 2 + 150 * np.cos(a)), int(h // 2 + 250 * np.sin(a)))
        cv2.circle(img, p, 18, (30, 100, 175), -1)

    # usure : zone legerement eclaircie et irreguliere
    wear = np.zeros((h, w), np.float32)
    cv2.ellipse(wear, (int(w * 0.72), int(h * 0.30)), (150, 110), 25, 0, 360, 1.0, -1)
    wear = cv2.GaussianBlur(wear, (0, 0), 45)
    img = np.clip(img.astype(np.float32) + wear[..., None] * 34, 0, 255).astype(np.uint8)

    # grain de laine
    img = np.clip(img.astype(np.float32) + rng.normal(0, 3.2, img.shape), 0, 255).astype(np.uint8)
    return img


def synth_rug_photo(scale: float = 1.0, seed: int = 7):
    """Le tapis 'photographie' : pose en perspective sur un fond, avec franges.
    Retourne (photo_bgr, masque_verite_terrain)."""
    rug = synth_rug(seed=seed)
    H, W = 1500, 2000
    photo = np.full((H, W, 3), 96, np.uint8)
    photo[:] = (120, 128, 138)

    src = np.float32([[0, 0], [rug.shape[1], 0], [rug.shape[1], rug.shape[0]], [0, rug.shape[0]]])
    dst = np.float32([[520, 180], [1490, 235], [1590, 1320], [430, 1265]])  # perspective legere
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(rug, M, (W, H))
    mask = cv2.warpPerspective(np.full(rug.shape[:2], 255, np.uint8), M, (W, H))

    # franges sur les deux petits cotes : irregulieres, c'est voulu
    rng = np.random.default_rng(seed + 1)
    for t in np.linspace(0.02, 0.98, 90):
        for edge_a, edge_b, out in ((dst[0], dst[1], -1), (dst[3], dst[2], 1)):
            p = edge_a + (edge_b - edge_a) * t
            L = rng.uniform(16, 30)
            q = p + np.array([rng.uniform(-4, 4), out * L])
            cv2.line(photo, tuple(p.astype(int)), tuple(q.astype(int)), (205, 210, 220), 2)
            cv2.line(mask, tuple(p.astype(int)), tuple(q.astype(int)), 255, 2)

    photo = np.where(mask[..., None] > 0, np.where(warped.any(axis=2, keepdims=True), warped, photo), photo)
    photo = np.clip(photo.astype(np.float32) + rng.normal(0, 2.0, photo.shape), 0, 255).astype(np.uint8)
    return photo, mask


def synth_room(w: int = 1600, h: int = 900, seed: int = 3):
    """Piece vide : mur, parquet en perspective, canape. Retourne (image, floor_quad)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    horizon = int(h * 0.42)
    img[:horizon] = (208, 212, 214)          # mur
    img[horizon:] = (110, 145, 178)          # parquet

    # lattes convergeant vers un point de fuite -> perspective credible
    vp = (int(w * 0.5), horizon - 120)
    for x in range(-w, 2 * w, 46):
        cv2.line(img, (x, h), vp, (92, 124, 156), 2)
    for i in range(14):
        y = int(horizon + (h - horizon) * (i / 14) ** 1.7)
        cv2.line(img, (0, y), (w, y), (100, 133, 166), 1)

    # gradient de lumiere : fenetre a gauche
    gx = np.linspace(1.22, 0.80, w, dtype=np.float32)[None, :]
    gy = np.linspace(1.06, 0.94, h, dtype=np.float32)[:, None]
    img = np.clip(img.astype(np.float32) * (gx * gy)[..., None], 0, 255).astype(np.uint8)

    # canape au fond
    cv2.rectangle(img, (int(w * 0.28), horizon - 60), (int(w * 0.72), horizon + 70), (150, 158, 168), -1)
    cv2.rectangle(img, (int(w * 0.28), horizon - 60), (int(w * 0.72), horizon - 10), (162, 170, 180), -1)

    img = np.clip(img.astype(np.float32) + rng.normal(0, 2.6, img.shape), 0, 255).astype(np.uint8)

    floor = np.float32([[int(w * 0.30), horizon + 78],
                        [int(w * 0.70), horizon + 78],
                        [int(w * 0.92), int(h * 0.97)],
                        [int(w * 0.08), int(h * 0.97)]])
    return img, floor
