"""Bibliotheque de decors reels.

Alternative a la generation : au lieu de fabriquer une piece a chaque image, on
photographie une fois un jeu de pieces VIDES et on y incruste les tapis. Cout
par image : zero. Photorealisme : total, puisque la piece est une photographie.

Deux risques majeurs du projet disparaissent avec cette approche :
  - "le decor genere passera-t-il pour une vraie photo ?" -- il en est une ;
  - "le modele de vision situe-t-il bien le plan du sol ?" -- il est calibre une
    fois par piece, a la main, et reutilise indefiniment.

Le prix a payer est la finitude : on ne dispose que des pieces photographiees.
La variation vient donc d'ailleurs -- choix de la piece, recadrage, position et
rotation du tapis au sol, etalonnage lumineux. C'est suffisant : deux images
d'un meme tapis doivent differer, elles n'ont pas besoin d'etre uniques au monde.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from babarug.geometry import FloorPlane, denormalize, floor_quad_plausibility

log = logging.getLogger(__name__)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class Scene:
    """Une piece vide photographiee, avec son plan du sol calibre."""

    scene_id: str
    image_path: Path
    floor_norm: tuple                 # 4 points (x, y) normalises, TL TR BR BL
    floor_width_m: float
    floor_depth_m: float
    interior_style: str
    room_type: str
    flooring: str = ""
    lighting: str = ""
    notes: str = ""
    # Marge de recadrage disponible : permet de varier le cadrage sans sortir
    # de la zone utile. 0 = la photo doit etre utilisee telle quelle.
    crop_margin: float = 0.08
    tags: list[str] = field(default_factory=list)

    def load(self) -> np.ndarray:
        img = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"decor illisible : {self.image_path}")
        return img

    def floor_plane(self, w: int, h: int) -> FloorPlane:
        return FloorPlane(denormalize(self.floor_norm, w, h),
                          self.floor_width_m, self.floor_depth_m)

    def to_json(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "image": self.image_path.name,
            "floor_norm": [list(p) for p in self.floor_norm],
            "floor_width_m": self.floor_width_m,
            "floor_depth_m": self.floor_depth_m,
            "interior_style": self.interior_style,
            "room_type": self.room_type,
            "flooring": self.flooring,
            "lighting": self.lighting,
            "notes": self.notes,
            "crop_margin": self.crop_margin,
            "tags": self.tags,
        }

    @classmethod
    def from_json(cls, d: dict, folder: Path) -> "Scene":
        return cls(
            scene_id=d["scene_id"],
            image_path=folder / d["image"],
            floor_norm=tuple(tuple(float(v) for v in p) for p in d["floor_norm"]),
            floor_width_m=float(d["floor_width_m"]),
            floor_depth_m=float(d["floor_depth_m"]),
            interior_style=d.get("interior_style", ""),
            room_type=d.get("room_type", ""),
            flooring=d.get("flooring", ""),
            lighting=d.get("lighting", ""),
            notes=d.get("notes", ""),
            crop_margin=float(d.get("crop_margin", 0.08)),
            tags=list(d.get("tags", [])),
        )

    def validate(self) -> list[str]:
        """Controles qui evitent de decouvrir un decor casse en pleine production."""
        problems: list[str] = []
        if not self.image_path.exists():
            problems.append(f"image absente : {self.image_path.name}")
            return problems
        img = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if img is None:
            problems.append("image illisible")
            return problems
        h, w = img.shape[:2]
        if min(w, h) < 900:
            problems.append(f"definition faible ({w}x{h}) : viser 2400px sur le grand cote")
        if len(self.floor_norm) != 4:
            problems.append("le plan du sol doit avoir exactement 4 points")
            return problems
        if not all(0.0 <= c <= 1.0 for p in self.floor_norm for c in p):
            problems.append("les points du sol doivent etre normalises entre 0 et 1")
        ok, geo = floor_quad_plausibility(denormalize(self.floor_norm, w, h), w, h)
        if not ok:
            problems += [f"plan du sol : {g}" for g in geo]
        if not (1.2 <= self.floor_width_m <= 12) or not (1.0 <= self.floor_depth_m <= 12):
            problems.append("dimensions du sol invraisemblables")
        return problems


class SceneLibrary:
    """Collection de decors, chargee depuis un dossier.

    Arborescence attendue :
        scenes/
        ├── salon_parisien_01.jpg
        ├── salon_parisien_01.json
        ├── salle_a_manger_sud_02.jpg
        └── salle_a_manger_sud_02.json
    """

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        self.scenes: list[Scene] = []
        self.problems: dict[str, list[str]] = {}
        if self.folder.exists():
            self._load()

    def _load(self):
        for meta in sorted(self.folder.glob("*.json")):
            try:
                s = Scene.from_json(json.loads(meta.read_text()), self.folder)
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                self.problems[meta.name] = [f"fiche illisible : {e}"]
                continue
            pb = s.validate()
            if pb:
                self.problems[s.scene_id] = pb
                log.warning("decor %s ecarte : %s", s.scene_id, "; ".join(pb))
                continue
            self.scenes.append(s)

    def __len__(self) -> int:
        return len(self.scenes)

    def styles(self) -> set[str]:
        return {s.interior_style for s in self.scenes}

    def rooms(self) -> set[str]:
        return {s.room_type for s in self.scenes}

    def matching(self, interior_style: str = "", room_type: str = "") -> list[Scene]:
        out = self.scenes
        if interior_style:
            out = [s for s in out if s.interior_style == interior_style] or out
        if room_type:
            out = [s for s in out if s.room_type == room_type] or out
        return out

    def pick(
        self, product_id: str, variant: str, interior_style: str = "",
        room_type: str = "", exclude: set[str] | None = None,
    ) -> Scene:
        """Choisit un decor de facon reproductible, en evitant les repetitions.

        Deterministe sur (product_id, variant) : relancer le meme tapis redonne
        les memes decors, ce qui rend les resultats comparables d'un essai a
        l'autre. Des tapis differents tombent sur des decors differents.
        """
        if not self.scenes:
            raise ValueError(
                f"bibliotheque de decors vide ({self.folder}). "
                "Ajouter des photos de pieces vides et les calibrer "
                "(voir scenes/README.md)."
            )
        pool = self.matching(interior_style, room_type)
        if exclude:
            pool = [s for s in pool if s.scene_id not in exclude] or pool
        h = hashlib.sha256(f"{product_id}|{variant}".encode()).hexdigest()
        return pool[int(h[:16], 16) % len(pool)]


def write_scene(folder: str | Path, scene: Scene) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{scene.scene_id}.json"
    path.write_text(json.dumps(scene.to_json(), ensure_ascii=False, indent=2))
    return path
