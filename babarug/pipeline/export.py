"""Etape 7 : EXPORT. Fichiers prets pour Shopify."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import cv2
import numpy as np


def slugify(text: str) -> str:
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    t = re.sub(r"[^a-zA-Z0-9]+", "-", t).strip("-").lower()
    return re.sub(r"-{2,}", "-", t)


def build_filename(template: str, **fields) -> str:
    """Nom de fichier configurable, oriente referencement.

    Exemple : "quelle-taille-tapis-{room}-baba-rug" -> quelle-taille-tapis-salon-baba-rug.jpg
    """
    name = template.format(**{k: slugify(str(v)) for k, v in fields.items()})
    return f"{slugify(name)}.jpg"


def export_jpeg(
    image_bgr: np.ndarray, path: str | Path, width: int = 1600, height: int = 900,
    max_kb: int | None = 400, min_quality: int = 68, start_quality: int = 92,
) -> dict:
    """Redimensionne en 16:9 et ecrit un JPEG sous un poids cible.

    Recherche dichotomique sur la qualite : on descend juste ce qu'il faut pour
    tenir le budget, et on s'arrete a `min_quality` plutot que de degrader le
    motif du tapis -- qui est le sujet de la photo.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    h, w = image_bgr.shape[:2]
    target = width / height
    if abs(w / h - target) > 0.01:
        if w / h > target:
            nw = int(h * target)
            image_bgr = image_bgr[:, (w - nw) // 2 : (w - nw) // 2 + nw]
        else:
            nh = int(w / target)
            top = int((h - nh) * 0.72)  # on garde le sol, pas le plafond
            image_bgr = image_bgr[top : top + nh, :]
    out = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_AREA)

    quality = start_quality
    if max_kb:
        lo, hi, best = min_quality, start_quality, None
        while lo <= hi:
            mid = (lo + hi) // 2
            ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, mid,
                                                 cv2.IMWRITE_JPEG_OPTIMIZE, 1])
            if not ok:
                break
            if len(buf) / 1024 <= max_kb:
                best, quality = mid, mid
                lo = mid + 1
            else:
                hi = mid - 1
        if best is None:
            quality = min_quality

    cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, quality, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
    kb = path.stat().st_size / 1024
    return {"path": str(path), "width": width, "height": height,
            "quality": quality, "kb": round(kb, 1),
            "over_budget": bool(max_kb and kb > max_kb)}
