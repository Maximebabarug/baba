"""Studio procedural : fabrication du decor par le code, sans modele ni API.

Le decor est construit, pas genere : une camera perspective reelle, un sol
texture en vue de dessus puis projete par homographie, un fond, une lumiere
directionnelle. Cout nul, resultat reproductible, et surtout -- le plan du sol
n'a pas a etre devine, il est connu exactement puisque c'est nous qui posons la
camera.

Le rendu vise le "studio produit" epure des sites de tapis haut de gamme : un
beau sol, un fond neutre, une lumiere qui sculpte. Ce n'est PAS un salon
haussmannien meuble, et il ne faut pas le vendre comme tel. En revanche il est
honnete, il ne montre jamais un meuble improbable, et il met le tapis au centre.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from babarug.geometry import FloorPlane

SENSOR_MM = 36.0  # plein format, pour convertir une focale en angle de champ


# ---------------------------------------------------------------- camera

@dataclass
class Camera:
    height_m: float = 1.45
    pitch_deg: float = 16.0      # positif = la camera regarde vers le bas
    focal_mm: float = 40.0
    roll_deg: float = 0.0

    def projector(self, width: int, height: int):
        """Retourne une fonction (X, Y) -> (u, v) projetant le plan du sol.

        Repere monde : le sol est le plan z = 0, la camera est en (0, 0, h) et
        regarde vers les Y croissants en plongeant de `pitch_deg`. X est la
        gauche-droite, Y l'eloignement.
        """
        f = (width / 2.0) / np.tan(np.radians(self._fov_h()) / 2.0)
        cx, cy = width / 2.0, height / 2.0
        t = np.radians(self.pitch_deg)
        sin_t, cos_t = np.sin(t), np.cos(t)
        h = self.height_m

        def project(X, Y):
            X = np.asarray(X, dtype=np.float64)
            Y = np.asarray(Y, dtype=np.float64)
            xc = X
            yc = Y * sin_t - h * cos_t
            zc = Y * cos_t + h * sin_t          # profondeur devant la camera
            zc = np.maximum(zc, 1e-6)
            return cx + f * xc / zc, cy - f * yc / zc

        return project

    def _fov_h(self) -> float:
        return float(np.degrees(2 * np.arctan(SENSOR_MM / (2 * self.focal_mm))))

    def floor_y_at_v(self, width: int, height: int, v: float) -> float:
        """Distance Y du point de sol qui se projette a l'ordonnee image `v`.

        Sert a savoir ou le sol entre dans le cadre. Poser la zone degagee "au
        juge" la faisait tomber sous le bord inferieur de l'image, et le tapis
        avec : le controle qualite renvoyait alors "tapis quasi absent".
        """
        f = (width / 2.0) / np.tan(np.radians(self._fov_h()) / 2.0)
        cy = height / 2.0
        t = np.radians(self.pitch_deg)
        sin_t, cos_t = np.sin(t), np.cos(t)
        h = self.height_m
        a = (cy - v) / f
        denom = sin_t - a * cos_t
        if abs(denom) < 1e-9:
            return float("inf")
        return float((a * h * sin_t + h * cos_t) / denom)

    def wall_line_v(self, width: int, height: int, wall_y_m: float) -> float:
        """Ordonnee image de la jonction sol/fond."""
        _, v = self.projector(width, height)(0.0, wall_y_m)
        return float(v)

    def floor_homography(self, width: int, height: int, x_m: float, y0_m: float, y1_m: float):
        """Homographie monde (X, Y) -> image, pour la dalle [-x/2, x/2] x [y0, y1]."""
        p = self.projector(width, height)
        world = np.float32([[-x_m / 2, y1_m], [x_m / 2, y1_m],
                            [x_m / 2, y0_m], [-x_m / 2, y0_m]])
        pts = np.float32([p(X, Y) for X, Y in world])
        return world, pts


# ---------------------------------------------------------------- textures de sol

def _fbm(shape, octaves: int, rng, persistence: float = 0.55) -> np.ndarray:
    """Bruit fractal simple : somme de bruits lisses de frequences croissantes."""
    h, w = shape
    out = np.zeros(shape, np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        s = 2 ** (o + 2)
        small = rng.random((max(2, h // (2 ** (octaves - o) * 2) + 2),
                            max(2, w // (2 ** (octaves - o) * 2) + 2))).astype(np.float32)
        out += amp * cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= persistence
    return out / max(total, 1e-6)


def _wood_base(rng, n: int, warm: float) -> np.ndarray:
    """Palette de lames.

    Un vrai parquet de chene varie surtout en CLARTE, tres peu en teinte : les
    lames restent toutes du chene. Tirer les trois canaux independamment donne
    un sol rose, vert et orange -- essaye, et immediatement faux. On tire donc
    un seul facteur de clarte par lame, plus un micro-decalage sur l'axe
    chaud/froid uniquement.
    """
    base = np.array([120, 150, 178], np.float32)      # BGR, chene clair
    base = base * np.array([1.0 - warm * 0.07, 1.0, 1.0 + warm * 0.07], np.float32)

    lum = 1.0 + rng.normal(0, 0.052, (n, 1)).astype(np.float32)      # clarte
    axis = rng.normal(0, 2.6, (n, 1)).astype(np.float32)             # chaud <-> froid
    cols = base[None, :] * lum + axis * np.array([-1.0, 0.0, 1.0], np.float32)[None, :]
    return np.clip(cols, 40, 240)


def parquet_texture(w_px: int, w_m: float, style: str = "point_de_hongrie",
                    seed: int = 0, warm: float = 0.0, h_m: float | None = None
                    ) -> np.ndarray:
    """Sol vu de dessus. `w_m` x `h_m` metres, a echelle ISOTROPE.

    On travaille en vue de dessus puis on projette : c'est ce qui donne une
    perspective juste sans dessiner quoi que ce soit en perspective.

    La texture doit couvrir une zone au bon rapport largeur/profondeur. Plaquer
    une texture carree sur une dalle rectangulaire etire le motif -- ici, un pas
    de lame grossi de 1.7x, et un point de Hongrie qui ne ressemble plus a du
    parquet.
    """
    rng = np.random.default_rng(seed)
    h_m = w_m if h_m is None else h_m
    m_per_px = w_m / w_px
    h_px = max(8, int(round(h_m / m_per_px)))
    yy, xx = np.mgrid[0:h_px, 0:w_px].astype(np.float32) * m_per_px  # en metres
    px = w_px

    if style == "travertin":
        return _stone_texture(w_px, h_px, m_per_px, rng, warm)
    if style == "beton":
        return _concrete_texture(w_px, h_px, rng)

    if style == "point_de_hongrie":
        # Lame de 11 cm de large, ~60 cm de long, coupee a 50 deg : l'emprise
        # horizontale d'une colonne vaut environ 60 x cos(50) = 42 cm. Une bande
        # de 62 cm donnait des V d'1,25 m -- du parquet de geant.
        pitch, angle_deg, band_m = 0.108, 50.0, 0.42
        band = np.floor(xx / band_m)
        sign = np.where(band % 2 == 0, 1.0, -1.0)
        s = yy + sign * xx * np.tan(np.radians(90 - angle_deg))
        plank_f = s / pitch
        plank_id = np.floor(plank_f)
        across = plank_f - plank_id
        uid = (plank_id * 131.0 + band * 977.0)
    else:  # lames droites, largeur 0.14 m, longueur 1.1 m
        pitch, length = 0.14, 1.10
        plank_id = np.floor(xx / pitch)
        across = xx / pitch - plank_id
        offset = (plank_id * 0.37) % 1.0
        seg = np.floor((yy / length) + offset)
        uid = plank_id * 131.0 + seg * 977.0
        # joint de bout
        end = np.abs(((yy / length + offset) - seg) - 0.0)
        across = np.where(end < 0.004, 0.0, across)

    shape = (h_px, w_px)
    n = 4096
    cols = _wood_base(rng, n, warm)
    idx = (np.abs(uid).astype(np.int64) * 2654435761) % n
    img = cols[idx]

    # veinage : bruit etire le long de la lame
    grain = _fbm(shape, 5, rng)
    k = max(3, int(px * 0.010) | 1)
    grain_a = cv2.GaussianBlur(grain, (1, k), 0)
    grain_b = cv2.GaussianBlur(grain, (k, 1), 0)
    if style == "point_de_hongrie":
        g = np.where(np.floor(xx / band_m) % 2 == 0, grain_a, grain_b)
        g = cv2.GaussianBlur(g, (3, 3), 0)
    else:
        g = grain_a
    img += ((g - g.mean()) * 17.0)[..., None]

    # joints entre lames : creux sombre, pas un trait vectoriel
    seam = np.clip(1.0 - np.minimum(across, 1.0 - across) / 0.042, 0, 1) ** 2
    img *= (1.0 - 0.20 * seam)[..., None]

    # variation lente d'ensemble + vernis
    slow = cv2.GaussianBlur(_fbm(shape, 3, rng), (0, 0), max(w_px, h_px) * 0.05)
    img *= (0.96 + 0.08 * (slow - slow.min()) / max(float(np.ptp(slow)), 1e-6))[..., None]
    img = cv2.GaussianBlur(img, (3, 3), 0)   # pas de joints au rasoir
    return np.clip(img, 0, 255).astype(np.uint8)


def _stone_texture(w_px, h_px, m_per_px, rng, warm) -> np.ndarray:
    yy, xx = np.mgrid[0:h_px, 0:w_px].astype(np.float32) * m_per_px
    tile = 0.60
    ix, iy = np.floor(xx / tile), np.floor(yy / tile)
    uid = (np.abs(ix * 131 + iy * 977).astype(np.int64) * 2654435761) % 512
    stone = np.array([176, 190, 203], np.float32)
    cols = np.clip(stone[None, :] * (1.0 + rng.normal(0, 0.035, (512, 1)).astype(np.float32))
                   + rng.normal(0, 2.0, (512, 1)).astype(np.float32)
                   * np.array([-1.0, 0.0, 1.0], np.float32)[None, :], 60, 250)
    img = cols[uid]
    veins = _fbm((h_px, w_px), 6, rng)
    img += ((veins - veins.mean()) * 16.0)[..., None]
    jx = np.minimum(xx / tile - ix, 1 - (xx / tile - ix))
    jy = np.minimum(yy / tile - iy, 1 - (yy / tile - iy))
    joint = np.clip(1.0 - np.minimum(jx, jy) / 0.018, 0, 1) ** 2
    img *= (1.0 - 0.18 * joint)[..., None]
    img[..., 2] *= 1.0 + warm * 0.06
    return np.clip(img, 0, 255).astype(np.uint8)


def _concrete_texture(w_px, h_px, rng) -> np.ndarray:
    img = np.full((h_px, w_px, 3), (158, 160, 162), np.float32)
    n = _fbm((h_px, w_px), 6, rng)
    img += ((n - n.mean()) * 26.0)[..., None]
    fine = rng.normal(0, 3.0, (h_px, w_px)).astype(np.float32)
    img += cv2.GaussianBlur(fine, (3, 3), 0)[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- styles

@dataclass
class StudioStyle:
    key: str
    label: str
    floor: str = "point_de_hongrie"
    floor_warm: float = 0.25
    wall_bgr: tuple = (222, 224, 224)
    wall_warm: float = 0.0
    light_from: float = -0.75      # -1 = lumiere a gauche, +1 = a droite
    light_strength: float = 0.30
    window_patch: bool = True
    patch_softness: float = 0.10
    plinthe: bool = True
    camera: Camera = field(default_factory=Camera)
    floor_span_m: float = 9.0
    wall_y_m: float = 6.2


STUDIO_STYLES: dict[str, StudioStyle] = {
    "parisien_chene": StudioStyle(
        "parisien_chene", "Parquet chêne, mur clair",
        floor="point_de_hongrie", floor_warm=0.30, wall_bgr=(224, 226, 226),
        light_from=-0.8, light_strength=0.32,
        camera=Camera(height_m=1.52, pitch_deg=12.5, focal_mm=38.0)),
    "atelier_chene_fonce": StudioStyle(
        "atelier_chene_fonce", "Chêne chaud, mur taupe",
        floor="point_de_hongrie", floor_warm=0.55, wall_bgr=(196, 199, 203),
        light_from=0.7, light_strength=0.38, window_patch=True,
        camera=Camera(height_m=1.66, pitch_deg=16.0, focal_mm=34.0)),
    "lames_larges": StudioStyle(
        "lames_larges", "Lames larges, mur écru",
        floor="lames", floor_warm=0.18, wall_bgr=(228, 229, 227),
        light_from=-0.55, light_strength=0.26,
        camera=Camera(height_m=1.38, pitch_deg=10.0, focal_mm=45.0)),
    "travertin_sud": StudioStyle(
        "travertin_sud", "Travertin, mur chaulé",
        floor="travertin", floor_warm=0.35, wall_bgr=(226, 230, 232), wall_warm=0.25,
        light_from=0.85, light_strength=0.42,
        camera=Camera(height_m=1.58, pitch_deg=14.5, focal_mm=33.0)),
    "beton_brut": StudioStyle(
        "beton_brut", "Béton ciré, mur gris",
        floor="beton", floor_warm=0.0, wall_bgr=(198, 199, 200),
        light_from=-0.9, light_strength=0.34, window_patch=True,
        camera=Camera(height_m=1.72, pitch_deg=18.0, focal_mm=30.0)),
}


# ---------------------------------------------------------------- rendu

def render_studio(
    width: int = 1600, height: int = 900, style: StudioStyle | str = "parisien_chene",
    seed: int = 0, clear_zone_m: tuple[float, float] = (3.4, 2.6),
) -> tuple[np.ndarray, FloorPlane]:
    """Fabrique un decor de studio et retourne (image, plan du sol EXACT).

    Le plan du sol n'est pas estime : la camera etant posee par nous, la zone
    degagee est projetee analytiquement. C'est la difference decisive avec un
    decor genere, ou il faut demander a un modele de vision ou se trouve le sol.
    """
    st = STUDIO_STYLES[style] if isinstance(style, str) else style
    rng = np.random.default_rng(seed)
    cam = st.camera
    proj = cam.projector(width, height)

    # --- sol : texture vue de dessus, projetee par homographie ---
    tex_px = 1500
    span = st.floor_span_m
    y_near, y_far = 1.4, st.wall_y_m
    tex = parquet_texture(tex_px, span, st.floor, seed=seed * 7 + 11,
                          warm=st.floor_warm, h_m=y_far - y_near)
    th, tw = tex.shape[:2]
    src = np.float32([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]])
    world = [(-span / 2, y_far), (span / 2, y_far), (span / 2, y_near), (-span / 2, y_near)]
    dst = np.float32([proj(X, Y) for X, Y in world])
    H = cv2.getPerspectiveTransform(src, dst)
    floor_img = cv2.warpPerspective(tex, H, (width, height), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)

    v_wall = cam.wall_line_v(width, height, st.wall_y_m)
    yy = np.arange(height, dtype=np.float32)[:, None]
    floor_mask = cv2.GaussianBlur(
        np.repeat((yy > v_wall).astype(np.float32), width, axis=1), (1, 3), 0)

    # --- fond : degrade vertical + micro-texture d'enduit ---
    wall = np.zeros((height, width, 3), np.float32)
    wall[:] = np.array(st.wall_bgr, np.float32) * np.array(
        [1 - st.wall_warm * 0.06, 1.0, 1 + st.wall_warm * 0.06], np.float32)
    grad = np.clip((yy / max(v_wall, 1.0)), 0, 1) ** 1.4
    wall *= (0.90 + 0.13 * grad)[..., None]
    plaster = cv2.GaussianBlur(rng.normal(0, 2.2, (height, width)).astype(np.float32), (5, 5), 0)
    wall += plaster[..., None]
    # Un mur peint n'est jamais un aplat : legeres inegalites d'enduit et halo
    # de la source lumineuse. Sans cela le fond trahit immediatement le rendu.
    slowwall = cv2.GaussianBlur(_fbm((height, width), 3, rng), (0, 0), width * 0.09)
    wall *= (0.97 + 0.06 * (slowwall - slowwall.min())
             / max(float(np.ptp(slowwall)), 1e-6))[..., None]
    # Halo de la source : TRES large et faible. Resserre, il devient une tache
    # blanche qui lit comme une fuite de lumiere et non comme un mur eclaire.
    hx = (np.arange(width, dtype=np.float32) / width - (0.5 + st.light_from * 0.45))
    halo = np.exp(-(hx[None, :] ** 2) / 0.55)
    wall *= (1.0 + 0.055 * (st.light_strength / 0.3) * halo)[..., None]

    img = wall * (1 - floor_mask[..., None]) + floor_img * floor_mask[..., None]

    # --- jonction sol/mur : occlusion ambiante, et plinthe optionnelle ---
    band = np.exp(-((yy - v_wall) / max(height * 0.055, 1.0)) ** 2)
    img *= (1.0 - 0.22 * band)[..., None]
    if st.plinthe:
        hph = max(6.0, height * 0.030)
        top = v_wall - hph
        pl = ((yy > top) & (yy <= v_wall)).astype(np.float32)
        pl = cv2.GaussianBlur(np.repeat(pl, width, axis=1), (1, 3), 0)
        face = np.array(st.wall_bgr, np.float32) * 1.05
        # legere ombre en haut de la plinthe, elle n'est pas dans le plan du mur
        shade = np.clip(1.0 - (yy - top) / max(hph, 1.0), 0, 1) ** 2
        img = img * (1 - pl[..., None]) + \
            (face[None, None, :] * (1.0 - 0.13 * shade)[..., None]) * pl[..., None]

    # --- lumiere : cle directionnelle + tache de fenetre au sol ---
    xx = np.arange(width, dtype=np.float32)[None, :]
    key = 1.0 + st.light_strength * (0.5 - np.abs(xx / width - (0.5 + st.light_from * 0.5)))
    img *= np.clip(key, 0.55, 1.5)[..., None]

    if st.window_patch:
        img = _window_patch(img, proj, st, rng, width, height)

    # --- profondeur de champ : le fond decroche legerement ---
    far = np.clip(1.0 - (yy - v_wall) / max(height * 0.45, 1.0), 0, 1) ** 2
    blurred = cv2.GaussianBlur(img, (0, 0), max(1.0, width / 900))
    img = img * (1 - far[..., None] * 0.75) + blurred * (far[..., None] * 0.75)

    # --- vignettage + grain photographique ---
    gx, gy = np.meshgrid(np.linspace(-1, 1, width, dtype=np.float32),
                         np.linspace(-1, 1, height, dtype=np.float32))
    img *= (1.0 - 0.14 * np.clip(gx ** 2 + gy ** 2, 0, 1.6) ** 1.4)[..., None]
    img += rng.normal(0, 1.9, img.shape).astype(np.float32)

    out = np.clip(img, 0, 255).astype(np.uint8)

    # --- zone degagee : bornee par le sol REELLEMENT visible, puis projetee ---
    cw, cd = clear_zone_m
    y_bottom = cam.floor_y_at_v(width, height, height - 1)
    y_lo = max(y_bottom * 1.02, y_near)
    y_hi = st.wall_y_m - 0.55
    if y_hi - y_lo < cd:
        cd = max(0.8, (y_hi - y_lo) * 0.92)
    centre = y_lo + (y_hi - y_lo - cd) * 0.42 + cd / 2
    corners = [(-cw / 2, centre + cd / 2), (cw / 2, centre + cd / 2),
               (cw / 2, centre - cd / 2), (-cw / 2, centre - cd / 2)]
    quad = np.float32([proj(X, Y) for X, Y in corners])
    return out, FloorPlane(quad, width_m=cw, depth_m=cd)


def _window_patch(img, proj, st: StudioStyle, rng, width, height):
    """Tache de lumiere de fenetre projetee au sol.

    Un aplat lumineux uniforme sent le rendu. Une flaque de soleil avec un bord
    net d'un cote et diffus de l'autre est le detail qui fait basculer l'oeil du
    cote de la photographie -- et elle coute trois lignes.
    """
    sign = 1.0 if st.light_from > 0 else -1.0
    x0 = sign * float(rng.uniform(0.3, 1.1))
    w = float(rng.uniform(1.1, 1.8))
    y0 = float(rng.uniform(0.9, 1.6))
    d = float(rng.uniform(1.4, 2.4))
    skew = sign * float(rng.uniform(0.3, 0.8))

    quad = np.float32([proj(X, Y) for X, Y in [
        (x0 - w / 2 + skew, y0 + d), (x0 + w / 2 + skew, y0 + d),
        (x0 + w / 2, y0), (x0 - w / 2, y0)]])
    m = np.zeros((height, width), np.float32)
    cv2.fillConvexPoly(m, quad.astype(np.int32), 1.0)
    k = max(3, int(min(width, height) * st.patch_softness) | 1)
    m = cv2.GaussianBlur(m, (k, k), 0)
    return img * (1.0 + m[..., None] * st.light_strength * 0.55)
