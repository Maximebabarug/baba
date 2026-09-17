"""ImageGenerationProvider : fabrication du DECOR.

Rappel d'architecture : ces providers ne generent JAMAIS le tapis. Ils
produisent une piece vide dont le sol est degage. Le tapis est ensuite incruste
geometriquement. C'est la raison pour laquelle le choix du generateur est un
parametre secondaire du projet et non son coeur : on peut en changer sans
toucher a la fidelite produit.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import time

import numpy as np
import requests

from babarug.models import SceneBrief
from babarug.providers.base import CostTracker, ProviderError

log = logging.getLogger(__name__)


def _to_png(data: bytes, width: int, height: int) -> bytes:
    """Normalise la sortie du generateur au format attendu (recadrage centre)."""
    from PIL import Image

    im = Image.open(io.BytesIO(data)).convert("RGB")
    target = width / height
    got = im.width / im.height
    if abs(got - target) > 0.01:
        if got > target:  # trop large -> rogner les cotes
            new_w = int(im.height * target)
            left = (im.width - new_w) // 2
            im = im.crop((left, 0, left + new_w, im.height))
        else:            # trop haut -> rogner le haut en priorite, garder le sol
            new_h = int(im.width / target)
            top = int((im.height - new_h) * 0.72)
            im = im.crop((0, top, im.width, top + new_h))
    im = im.resize((width, height), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


class GeminiImageProvider(CostTracker):
    """Google Gemini (famille "Nano Banana"). Bon rendu d'interieur, rapide.

    Verifier le nom de modele et le endpoint avant mise en production : cette
    famille bouge vite. La variable d'environnement BABARUG_GEMINI_MODEL permet
    d'en changer sans toucher au code.
    """

    ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    PRICE_PER_IMAGE = 0.134

    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: int = 120):
        super().__init__()
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = model or os.environ.get("BABARUG_GEMINI_MODEL", "gemini-3-pro-image-preview")
        self.timeout = timeout
        if not self.api_key:
            raise ProviderError("GEMINI_API_KEY absent")

    def generate_scene(self, brief: SceneBrief, width: int, height: int, seed: int) -> bytes:
        body = {
            "contents": [{"parts": [{"text": brief.prompt()}]}],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {"aspectRatio": "16:9"},
            },
        }
        r = requests.post(
            self.ENDPOINT.format(model=self.model),
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            json=body,
            timeout=self.timeout,
        )
        if r.status_code >= 400:
            raise ProviderError(f"Gemini {r.status_code}: {r.text[:300]}")
        try:
            parts = r.json()["candidates"][0]["content"]["parts"]
            raw = next(
                base64.b64decode(p["inlineData"]["data"]) for p in parts if "inlineData" in p
            )
        except (KeyError, IndexError, StopIteration) as e:
            raise ProviderError(f"reponse Gemini inattendue : {r.text[:300]}") from e
        self._charge("generate_scene", self.PRICE_PER_IMAGE, brief.slug)
        return _to_png(raw, width, height)


class FluxKontextProvider(CostTracker):
    """Black Forest Labs FLUX.1 Kontext. Rapide, tres bon en edition guidee.

    Interessant surtout pour la passe d'edition locale (harmonisation), ou sa
    stabilite geometrique aide.
    """

    BASE = "https://api.bfl.ai/v1"
    PRICE_PER_IMAGE = 0.04

    def __init__(self, api_key: str | None = None, endpoint: str = "flux-kontext-pro",
                 timeout: int = 180):
        super().__init__()
        self.api_key = api_key or os.environ.get("BFL_API_KEY", "")
        self.endpoint = endpoint
        self.timeout = timeout
        if not self.api_key:
            raise ProviderError("BFL_API_KEY absent")

    def _submit_and_poll(self, payload: dict) -> bytes:
        h = {"x-key": self.api_key, "Content-Type": "application/json"}
        r = requests.post(f"{self.BASE}/{self.endpoint}", headers=h, json=payload, timeout=60)
        if r.status_code >= 400:
            raise ProviderError(f"BFL {r.status_code}: {r.text[:300]}")
        poll_url = r.json().get("polling_url")
        if not poll_url:
            raise ProviderError("BFL n'a pas renvoye de polling_url")

        deadline = time.time() + self.timeout
        while time.time() < deadline:
            time.sleep(1.5)
            p = requests.get(poll_url, headers=h, timeout=30).json()
            status = p.get("status")
            if status == "Ready":
                url = p["result"]["sample"]
                return requests.get(url, timeout=60).content
            if status in {"Error", "Failed", "Content Moderated", "Request Moderated"}:
                raise ProviderError(f"BFL status={status}: {str(p)[:300]}")
        raise ProviderError("BFL : delai depasse")

    def generate_scene(self, brief: SceneBrief, width: int, height: int, seed: int) -> bytes:
        raw = self._submit_and_poll(
            {"prompt": brief.prompt(), "aspect_ratio": "16:9", "seed": seed,
             "output_format": "png", "prompt_upsampling": False}
        )
        self._charge("generate_scene", self.PRICE_PER_IMAGE, brief.slug)
        return _to_png(raw, width, height)

    def edit(self, image_png: bytes, mask_png: bytes, prompt: str, refs=None) -> bytes:
        """Retouche guidee. ATTENTION : l'appelant doit garantir que le masque
        exclut le tapis. Le QC verifie ensuite que le produit n'a pas bouge."""
        raw = self._submit_and_poll(
            {"prompt": prompt, "input_image": base64.b64encode(image_png).decode(),
             "output_format": "png"}
        )
        self._charge("edit", self.PRICE_PER_IMAGE, prompt[:40])
        return raw


class OfflineSceneProvider(CostTracker):
    """Generateur procedural, sans API, sans cout, deterministe.

    Sert a trois choses concretes :
      - faire tourner le pipeline complet en CI, sans cle ni facture ;
      - isoler un bug de geometrie d'un alea de generateur ;
      - demontrer le pipeline avant d'avoir souscrit a quoi que ce soit.
    Le rendu est volontairement schematique : ce n'est pas un livrable client.
    """

    def __init__(self, **_):
        super().__init__()

    def generate_scene(self, brief: SceneBrief, width: int, height: int, seed: int) -> bytes:
        import cv2
        import numpy as np

        rng = np.random.default_rng(seed)
        img = np.zeros((height, width, 3), np.uint8)
        horizon = int(height * (0.38 + 0.08 * (brief.camera_height_m - 1.2)))
        horizon = int(np.clip(horizon, height * 0.25, height * 0.55))

        palettes = {
            "parisien contemporain": ((206, 210, 212), (108, 143, 176)),
            "haussmannien": ((212, 216, 220), (96, 132, 168)),
            "mediterraneen": ((214, 222, 226), (150, 168, 182)),
            "japandi": ((218, 220, 216), (150, 176, 196)),
        }
        wall, floor_col = palettes.get(brief.interior_style.lower(), ((205, 208, 210), (112, 146, 178)))
        img[:horizon] = wall
        img[horizon:] = floor_col

        vp = (int(width * 0.5), horizon - int(height * 0.14))
        for x in range(-width, 2 * width, max(28, width // 36)):
            cv2.line(img, (x, height), vp, tuple(int(c * 0.86) for c in floor_col), 2)
        for i in range(16):
            y = int(horizon + (height - horizon) * (i / 16) ** 1.7)
            cv2.line(img, (0, y), (width, y), tuple(int(c * 0.93) for c in floor_col), 1)

        # assise principale au fond, laisse le sol degage devant
        cv2.rectangle(img, (int(width * 0.26), horizon - int(height * 0.08)),
                      (int(width * 0.74), horizon + int(height * 0.09)), (152, 160, 170), -1)
        cv2.rectangle(img, (int(width * 0.26), horizon - int(height * 0.08)),
                      (int(width * 0.74), horizon - int(height * 0.01)), (166, 174, 184), -1)

        gx = np.linspace(1.20, 0.82, width, dtype=np.float32)[None, :]
        gy = np.linspace(1.05, 0.95, height, dtype=np.float32)[:, None]
        img = np.clip(img.astype(np.float32) * (gx * gy)[..., None], 0, 255).astype(np.uint8)
        img = np.clip(img.astype(np.float32) + rng.normal(0, 2.4, img.shape), 0, 255).astype(np.uint8)

        self._charge("generate_scene", 0.0, "offline")
        return cv2.imencode(".png", img)[1].tobytes()

    def default_floor(self, width: int, height: int, brief: SceneBrief):
        """Le sol est connu par construction : pas besoin d'un modele pour le lire."""
        from babarug.providers.base import FloorEstimate

        horizon = int(height * (0.38 + 0.08 * (brief.camera_height_m - 1.2)))
        horizon = int(np.clip(horizon, height * 0.25, height * 0.55)) if False else int(
            max(height * 0.25, min(height * 0.55, horizon))
        )
        y0 = (horizon + int(height * 0.10)) / height
        return FloorEstimate(
            quad_norm=((0.30, y0), (0.70, y0), (0.93, 0.97), (0.07, 0.97)),
            width_m=4.2, depth_m=3.4, confidence=1.0, notes="sol procedural connu",
        )
