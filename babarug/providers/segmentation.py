"""SegmentationProvider : isoler le tapis de son decor.

Etape la plus fragile du pipeline sur des photos non controlees, et il faut le
dire : quand le tapis a un champ ivoire et qu'il est photographie sur un mur
creme -- cas reel de BABA-RUG-0001 -- aucune segmentation automatique ne trouve
les franges, qui sont creme sur creme. La bonne reponse n'est pas un meilleur
modele, c'est un fond de prise de vue contrastant. Voir ARCHITECTURE.md.

Deux implementations :
  LocalGrabCut  -- sans API, sans cout, correct sur fond contraste
  RemoteMatting -- SAM 3 / BiRefNet / RMBG derriere un endpoint HTTP
"""

from __future__ import annotations

import io
import os

import cv2
import numpy as np
import requests

from babarug.providers.base import CostTracker, ProviderError


def _refine_edges(mask: np.ndarray, image_bgr: np.ndarray, radius: int = 3) -> np.ndarray:
    """Recolle le masque aux contours reels sans lisser les irregularites.

    Un filtre guide suit les aretes de l'image : les franges et les bords
    ondules survivent, la ou une simple ouverture morphologique les raboterait.
    """
    m = mask.astype(np.float32) / 255.0
    guide = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    try:
        refined = cv2.ximgproc.guidedFilter(guide, m, radius * 4, 1e-3)
    except (AttributeError, cv2.error):
        refined = cv2.bilateralFilter(m, radius * 4 + 1, 0.1, radius * 4)
    return (np.clip(refined, 0, 1) * 255).astype(np.uint8)


class LocalGrabCutProvider(CostTracker):
    """GrabCut initialise automatiquement. Gratuit, hors ligne, deterministe.

    Limites assumees : echoue quand le tapis et le fond ont la meme valeur, et
    perd les franges fines. Suffisant pour un fond contraste et pour la CI.
    """

    def __init__(self, iterations: int = 5, border_frac: float = 0.06, **_):
        super().__init__()
        self.iterations = iterations
        self.border_frac = border_frac

    def segment_rug(self, image_bgr: np.ndarray, rect=None) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        scale = min(1.0, 900 / max(h, w))
        small = cv2.resize(image_bgr, None, fx=scale, fy=scale) if scale < 1 else image_bgr
        sh, sw = small.shape[:2]

        if rect is None:
            b = self.border_frac
            rect = (int(sw * b), int(sh * b), int(sw * (1 - 2 * b)), int(sh * (1 - 2 * b)))
        else:
            x, y, rw, rh = rect
            rect = (int(x * scale), int(y * scale), int(rw * scale), int(rh * scale))

        mask = np.zeros((sh, sw), np.uint8)
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(small, mask, rect, bgd, fgd, self.iterations, cv2.GC_INIT_WITH_RECT)
        except cv2.error as e:
            raise ProviderError(f"GrabCut a echoue : {e}") from e

        out = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        if scale < 1:
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)
        out = _refine_edges(out, image_bgr)
        self._charge("segment_rug", 0.0, "local")
        return out


class RemoteMattingProvider(CostTracker):
    """Segmentation hebergee (SAM 3, BiRefNet, RMBG-2.0...).

    L'endpoint est configurable : le pipeline ne depend pas d'un fournisseur.
    Attendu : POST multipart {file} -> PNG avec canal alpha, ou masque 8 bits.
    """

    def __init__(self, endpoint: str | None = None, api_key: str | None = None,
                 price: float = 0.003, timeout: int = 90, **_):
        super().__init__()
        self.endpoint = endpoint or os.environ.get("BABARUG_MATTING_ENDPOINT", "")
        self.api_key = api_key or os.environ.get("BABARUG_MATTING_KEY", "")
        self.price = price
        self.timeout = timeout
        if not self.endpoint:
            raise ProviderError("BABARUG_MATTING_ENDPOINT absent")

    def segment_rug(self, image_bgr: np.ndarray, rect=None) -> np.ndarray:
        ok, buf = cv2.imencode(".png", image_bgr)
        if not ok:
            raise ProviderError("encodage PNG impossible")
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = {"prompt": "rug carpet including its fringes and irregular edges"}
        r = requests.post(
            self.endpoint, headers=headers, files={"file": ("rug.png", buf.tobytes(), "image/png")},
            data=data, timeout=self.timeout,
        )
        if r.status_code >= 400:
            raise ProviderError(f"matting {r.status_code}: {r.text[:200]}")

        from PIL import Image

        im = Image.open(io.BytesIO(r.content))
        arr = np.array(im)
        if arr.ndim == 3 and arr.shape[2] == 4:
            mask = arr[:, :, 3]
        elif arr.ndim == 2:
            mask = arr
        else:
            mask = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        if mask.shape[:2] != image_bgr.shape[:2]:
            mask = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]))
        self._charge("segment_rug", self.price, "remote")
        return (mask > 127).astype(np.uint8) * 255


def mask_quality(mask: np.ndarray, image_bgr: np.ndarray) -> tuple[float, list[str]]:
    """Note la qualite d'un masque AVANT de s'en servir.

    Un mauvais masque contamine tout l'aval : mieux vaut le detecter ici et
    demander une reprise humaine que produire deux lifestyles inexploitables.
    """
    problems: list[str] = []
    h, w = mask.shape[:2]
    area = float((mask > 127).sum()) / (h * w)
    if area < 0.06:
        problems.append(f"masque minuscule ({area:.1%} de l'image) : tapis non trouve ?")
    if area > 0.96:
        problems.append("masque quasi plein cadre : le fond a ete inclus")

    n, _, stats, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8), 8)
    if n > 1:
        areas = np.sort(stats[1:, cv2.CC_STAT_AREA])[::-1]
        if len(areas) > 1 and areas[1] > 0.10 * areas[0]:
            problems.append("plusieurs zones detachees : le detourage a pris du decor")

    # Contraste entre le tapis et ce qui l'entoure : c'est ce qui conditionne
    # la reussite du detourage des franges.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    outside = (cv2.dilate(mask, k) > 127) & (mask <= 127)
    inside = mask > 127
    contrast = 1.0
    if outside.any() and inside.any():
        g = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        contrast = abs(float(g[inside].mean()) - float(g[outside].mean())) / 255.0
        if contrast < 0.09:
            problems.append(
                f"contraste tapis/fond tres faible ({contrast:.2f}) : les franges seront "
                "perdues. Rephotographier sur un fond contrastant."
            )
    score = max(0.0, min(1.0, 1.0 - 0.3 * len(problems))) * min(1.0, contrast / 0.18)
    return score, problems
