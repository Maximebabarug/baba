"""Chaine complete pour UN tapis : import -> DNA -> plate -> 2 lifestyles -> export.

C'est le point d'entree unique utilise par la CLI et par l'interface web.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from babarug.models import RugDNA, Verdict
from babarug.pipeline.export import build_filename, export_jpeg
from babarug.pipeline.loop import RenderSettings, render_variant
from babarug.pipeline.fringes import recover_fringes
from babarug.pipeline.plate import extract_plate, pick_plate_photo
from babarug.providers.registry import get_generation, get_segmentation, get_vision
from babarug.providers.segmentation import mask_quality
from babarug.styles import build_brief, surprise_me

log = logging.getLogger(__name__)

PHOTO_EXT = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class RunConfig:
    out_root: Path = Path("data")
    scenes_folder: str = "scenes"
    vision: str | None = None
    generation: str | None = None
    segmentation: str | None = None
    styles: list[tuple[str, str]] | None = None   # None = "surprends-moi"
    filename_template: str = "{pid}-{room}-tapis-baba-rug"
    max_kb: int = 400
    settings: RenderSettings = field(default_factory=RenderSettings)
    reuse_dna: bool = True


@dataclass
class RunReport:
    product_id: str
    dna: RugDNA | None
    results: list = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    seconds: float = 0.0
    needs_human: bool = False

    @property
    def verdicts(self) -> list[str]:
        return [r.qc.verdict.value for r in self.results]


def load_photos(folder: Path) -> list[tuple[str, bytes]]:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in PHOTO_EXT)
    if not files:
        raise FileNotFoundError(f"aucune photo dans {folder}")
    return [(p.name, p.read_bytes()) for p in files]


def run_product(
    product_id: str, photo_dir: Path, cfg: RunConfig | None = None,
    declared_size_cm=None, on_step=None,
) -> RunReport:
    cfg = cfg or RunConfig()
    t0 = time.time()
    rep = RunReport(product_id=product_id, dna=None)
    step = on_step or (lambda m: log.info(m))

    out_dir = cfg.out_root / product_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Le provider de vision n'est construit QUE s'il sert : analyser des photos,
    # situer un sol non calibre, ou donner un avis de scene. Avec un RUG DNA
    # deja etabli et une bibliotheque de decors calibres, la chaine complete
    # tourne sans aucune cle d'API -- et l'instancier d'office la cassait.
    _vision_cache: list = []

    def vision_lazy():
        if not _vision_cache:
            _vision_cache.append(get_vision(cfg.vision))
        return _vision_cache[0]

    class _LazyVision:
        """Fait patienter la construction jusqu'au premier appel reel."""

        def __getattr__(self, name):
            return getattr(vision_lazy(), name)

        @property
        def costs(self):
            return _vision_cache[0].costs if _vision_cache else []

    vision = _LazyVision()
    gen = get_generation(cfg.generation, folder=cfg.scenes_folder, product_id=product_id)
    seg = get_segmentation(cfg.segmentation)

    # ---------------------------------------------------------------- 1. DNA
    dna_path = out_dir / "rug_dna.json"
    step("1/6 analyse des photos")
    photos = load_photos(photo_dir)
    if cfg.reuse_dna and dna_path.exists():
        dna = RugDNA.model_validate(json.loads(dna_path.read_text()))
        step(f"    RUG DNA reutilise ({dna.n_photos} photos)")
    else:
        dna = vision_lazy().analyze_rug(product_id, photos, declared_size_cm)
        dna_path.write_text(dna.model_dump_json(indent=2))
    rep.dna = dna
    if dna.needs_human_check:
        rep.needs_human = True
        rep.warnings.append(
            "RUG DNA incertain (ratio peu fiable, forme irreguliere ou aucune photo "
            "utilisable comme plate) : validation humaine recommandee avant generation"
        )

    # ---------------------------------------------------------------- 2. plate
    step("2/6 detourage et redressement du tapis")
    plate_name = dna.plate_photo if any(n == dna.plate_photo for n, _ in photos) \
        else pick_plate_photo(dna)
    raw = next(d for n, d in photos if n == plate_name)
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"photo illisible : {plate_name}")

    mask = seg.segment_rug(img)

    # Les franges sont systematiquement perdues par un detourage generique quand
    # elles sont de la couleur du fond. On tente de les recuperer avant de juger
    # le masque : sans elles, la plate coupe le tapis net et le rendu final
    # montre un tapis sans franges -- alteration produit inacceptable.
    if dna.fringe and "aucune" not in dna.fringe.lower():
        fr = recover_fringes(img, mask)
        rep.warnings += [f"franges : {n}" for n in fr.notes]
        if fr.found:
            gain = (fr.mask > 127).sum() / max((mask > 127).sum(), 1) - 1
            step(f"    franges recuperees (+{gain:.1%} de surface, "
                 f"separation {fr.separation:.1f} ecart-type)")
            mask = fr.mask
        else:
            rep.needs_human = True
            rep.warnings.append(
                "FRANGES NON RECUPEREES : le RUG DNA en decrit, le detourage n'en "
                "trouve pas. Le rendu montrera un tapis coupe net."
            )

    score, problems = mask_quality(mask, img)
    for p in problems:
        rep.warnings.append(f"detourage ({plate_name}) : {p}")
    if score < 0.55:
        rep.needs_human = True
        rep.warnings.append(
            f"qualite du detourage insuffisante ({score:.2f}) : le masque doit etre "
            "repris a la main, sinon les deux lifestyles seront faux"
        )

    plate = extract_plate(img, mask, dna.aspect_ratio)
    rep.warnings += [f"plate : {w}" for w in plate.warnings]
    cv2.imwrite(str(out_dir / "plate.png"), plate.rgba)

    # ---------------------------------------------------------------- 3. styles
    combos = cfg.styles or surprise_me(product_id, 2)
    step(f"3/6 styles retenus : {combos}")

    # ---------------------------------------------------------------- 4-6.
    for i, (style_key, room_key) in enumerate(combos):
        variant = chr(ord("A") + i)
        step(f"4/6 generation lifestyle {variant} ({style_key} / {room_key})")
        brief = build_brief(product_id, style_key, room_key, variant)
        result, image, warns = render_variant(
            dna, plate, brief, variant, gen, vision,
            qc_provider=vision, settings=cfg.settings,
            seed=abs(hash((product_id, variant))) % 100000, on_step=step,
        )
        rep.warnings += warns

        raw_path = out_dir / f"lifestyle_{variant}.png"
        cv2.imwrite(str(raw_path), image)
        result.image_path = str(raw_path)

        step(f"5/6 controle qualite {variant} : {result.qc.verdict.value}")
        if result.qc.verdict != Verdict.REJECTED:
            fn = build_filename(cfg.filename_template, pid=product_id, room=room_key,
                                style=style_key, variant=variant)
            info = export_jpeg(image, out_dir / "export" / fn,
                               cfg.settings.width, cfg.settings.height, cfg.max_kb)
            result.export_path = info["path"]
            if info["over_budget"]:
                rep.warnings.append(f"{fn} depasse le budget de poids ({info['kb']} ko)")
        else:
            rep.needs_human = True
            rep.warnings.append(f"lifestyle {variant} REJETE : pas d'export")
        rep.results.append(result)

    step("6/6 termine")
    rep.cost_usd = round(
        sum(c.usd for p in (_vision_cache + [gen, seg]) for c in getattr(p, "costs", [])), 4
    )
    rep.seconds = round(time.time() - t0, 1)
    if any(r.qc.verdict != Verdict.APPROVED for r in rep.results):
        rep.needs_human = True

    (out_dir / "report.json").write_text(json.dumps({
        "product_id": product_id,
        "verdicts": rep.verdicts,
        "cost_usd": rep.cost_usd,
        "seconds": rep.seconds,
        "needs_human": rep.needs_human,
        "warnings": rep.warnings,
        "results": [json.loads(r.model_dump_json()) for r in rep.results],
    }, ensure_ascii=False, indent=2))
    return rep


def run_batch(root: Path, cfg: RunConfig | None = None, on_step=None) -> list[RunReport]:
    """Traite un dossier de shooting : un sous-dossier par tapis.

    Repond directement au besoin : apres un shooting, on depose N dossiers et on
    lance. Un tapis qui echoue n'interrompt pas le lot -- c'est la difference
    entre une demo et un outil de production.
    """
    reports: list[RunReport] = []
    dirs = sorted(d for d in root.iterdir() if d.is_dir())
    for i, d in enumerate(dirs, 1):
        pid = d.name
        try:
            if on_step:
                on_step(f"[{i}/{len(dirs)}] {pid}")
            reports.append(run_product(pid, d, cfg, on_step=on_step))
        except Exception as e:
            log.exception("echec sur %s", pid)
            r = RunReport(product_id=pid, dna=None, needs_human=True)
            r.warnings.append(f"ECHEC : {type(e).__name__}: {e}")
            reports.append(r)
    return reports
