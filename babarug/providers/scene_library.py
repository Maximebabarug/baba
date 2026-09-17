"""ImageGenerationProvider adosse a une bibliotheque de decors REELS.

Implemente la meme interface que les generateurs distants : le pipeline ne voit
aucune difference et n'a pas une ligne a changer. La bibliotheque est
simplement un fournisseur de decors qui ne facture rien et ne se trompe jamais
sur le photorealisme.

`default_floor` est fourni : le plan du sol de chaque decor etant calibre une
fois pour toutes, aucun appel de vision n'est necessaire pour situer le tapis.
"""

from __future__ import annotations

import hashlib
import logging

import cv2
import numpy as np

from babarug.models import SceneBrief
from babarug.providers.base import CostTracker, FloorEstimate, ProviderError
from babarug.scenes import Scene, SceneLibrary

log = logging.getLogger(__name__)


def _rng(*parts) -> np.random.Generator:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return np.random.default_rng(int(h[:16], 16))


class SceneLibraryProvider(CostTracker):
    """Sert des photographies de pieces vides au lieu d'en generer.

    La variation entre deux images d'un meme tapis vient de quatre leviers, tous
    gratuits : le choix du decor, le recadrage, l'etalonnage lumineux, et --
    applique plus loin dans le pipeline -- la position du tapis au sol. C'est
    assez pour que deux lifestyles se distinguent nettement sans disposer d'un
    nombre infini de pieces.
    """

    def __init__(self, folder: str = "scenes", product_id: str = "", **_):
        super().__init__()
        self.library = SceneLibrary(folder)
        self.product_id = product_id
        self._used: set[str] = set()
        self._last: dict[str, tuple[Scene, np.ndarray, tuple]] = {}
        if len(self.library) == 0:
            raise ProviderError(
                f"bibliotheque de decors vide ({folder}). "
                "Deposer des photos de pieces vides puis les calibrer : "
                "python -m babarug.cli scene-add <photo>"
            )

    # ---------------------------------------------------------------- rendu
    def generate_scene(self, brief: SceneBrief, width: int, height: int, seed: int) -> bytes:
        scene = self.library.pick(
            self.product_id or brief.slug, brief.slug,
            interior_style=brief.style_key, room_type=brief.room_key,
            exclude=self._used,
        )
        self._used.add(scene.scene_id)

        img = scene.load()
        img, floor_norm = self._reframe(img, scene, width, height, seed)
        img = self._grade(img, brief, seed)

        self._last[brief.slug] = (scene, img, floor_norm)
        self._charge("generate_scene", 0.0, scene.scene_id)
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            raise ProviderError("encodage du decor impossible")
        return buf.tobytes()

    def default_floor(self, width: int, height: int, brief: SceneBrief) -> FloorEstimate:
        """Le plan du sol est connu : calibre une fois, jamais devine."""
        entry = self._last.get(brief.slug)
        if entry is None:
            raise ProviderError("default_floor appele avant generate_scene")
        scene, _, floor_norm = entry
        return FloorEstimate(
            quad_norm=floor_norm,
            width_m=scene.floor_width_m,
            depth_m=scene.floor_depth_m,
            confidence=1.0,
            notes=f"plan calibre du decor {scene.scene_id}",
        )

    # ---------------------------------------------------------------- variation
    def _reframe(self, img: np.ndarray, scene: Scene, width: int, height: int, seed):
        """Recadre en 16:9 avec un leger decentrement, et suit le plan du sol.

        Le quad du sol est exprime en coordonnees normalisees : il DOIT etre
        recalcule apres recadrage, sinon le tapis se retrouve pose a cote de la
        zone prevue. C'est le piege classique de ce genre de traitement.
        """
        rng = _rng(scene.scene_id, seed)
        h, w = img.shape[:2]
        target = width / height

        m = float(np.clip(scene.crop_margin, 0.0, 0.25))
        zoom = 1.0 - rng.uniform(0.0, m)          # 1.0 = pleine image
        cw, ch = w * zoom, h * zoom
        if cw / ch > target:
            cw = ch * target
        else:
            ch = cw / target

        # Decentrement horizontal libre, vertical volontairement biaise vers le
        # bas : c'est le sol qui interesse, pas le plafond.
        max_dx, max_dy = (w - cw) / 2, (h - ch) / 2
        dx = rng.uniform(-max_dx, max_dx) if max_dx > 1 else 0.0
        dy = rng.uniform(-0.25 * max_dy, max_dy) if max_dy > 1 else 0.0
        x0 = float(np.clip(w / 2 + dx - cw / 2, 0, w - cw))
        y0 = float(np.clip(h / 2 + dy - ch / 2, 0, h - ch))

        crop = img[int(y0):int(y0 + ch), int(x0):int(x0 + cw)]
        out = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)

        floor_norm = tuple(
            (float(np.clip((px * w - x0) / cw, -0.5, 1.5)),
             float(np.clip((py * h - y0) / ch, -0.5, 1.5)))
            for px, py in scene.floor_norm
        )
        return out, floor_norm

    def _grade(self, img: np.ndarray, brief: SceneBrief, seed: int) -> np.ndarray:
        """Etalonnage leger : temperature, exposition, vignettage.

        Volontairement DISCRET. Un etalonnage marque ferait deriver les couleurs
        de la piece, et donc -- apres relighting -- celles du tapis. On cherche a
        distinguer deux images, pas a les styliser.
        """
        rng = _rng(brief.slug, seed, "grade")
        f = img.astype(np.float32)
        f *= float(rng.uniform(0.95, 1.06))                       # exposition
        warm = float(rng.uniform(-0.035, 0.035))                  # temperature
        f[:, :, 0] *= 1.0 - warm                                  # bleu
        f[:, :, 2] *= 1.0 + warm                                  # rouge

        h, w = img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w * float(rng.uniform(0.4, 0.6)), h * float(rng.uniform(0.35, 0.6))
        r = np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)
        f *= (1.0 - float(rng.uniform(0.04, 0.12)) * (r / r.max()) ** 2)[..., None]
        return np.clip(f, 0, 255).astype(np.uint8)
