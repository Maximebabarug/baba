"""CLI Baba Rug Visual Studio.

  babarug run    BABA-RUG-0001 photos/          un tapis
  babarug batch  shooting_2026_09/              tout un shooting
  babarug dna    BABA-RUG-0001 photos/          analyse seule
  babarug doctor                                verifie l'installation
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from babarug.models import Verdict
from babarug.pipeline.loop import RenderSettings
from babarug.run import RunConfig, load_photos, run_batch, run_product

GREEN, YELLOW, RED, DIM, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[0m"
COLOR = {Verdict.APPROVED.value: GREEN, Verdict.REVIEW_REQUIRED.value: YELLOW,
         Verdict.REJECTED.value: RED}


def _cfg(a) -> RunConfig:
    styles = None
    if a.style and a.room:
        styles = [tuple(s.split("/")) for s in a.style] if "/" in a.style[0] else \
                 list(zip(a.style, a.room))
    return RunConfig(
        out_root=Path(a.out), scenes_folder=getattr(a, "scenes", "scenes"),
        vision=a.vision, generation=a.generation,
        segmentation=a.segmentation, styles=styles, max_kb=a.max_kb,
        filename_template=a.filename,
        settings=RenderSettings(width=a.width, height=a.height,
                                max_iterations=a.max_iterations,
                                review_with_model=not a.no_review),
    )


def _print(rep):
    print(f"\n  {rep.product_id}   {rep.seconds}s   ${rep.cost_usd:.3f}")
    for r in rep.results:
        c = COLOR.get(r.qc.verdict.value, "")
        print(f"    {c}{r.qc.verdict.value:16}{RESET} {r.variant}  "
              f"{r.brief.slug:38} {r.qc.fidelity.summary}  ({r.iterations} it.)")
        for f in r.qc.failures[:4]:
            print(f"        {DIM}{f}{RESET}")
        if r.export_path:
            print(f"        -> {r.export_path}")
    for w in rep.warnings:
        print(f"    {YELLOW}!{RESET} {w}")
    if rep.needs_human:
        print(f"    {YELLOW}>> relecture humaine requise{RESET}")


def cmd_run(a) -> int:
    size = tuple(float(x) for x in a.size.split("x")) if a.size else None
    rep = run_product(a.product_id, Path(a.photos), _cfg(a), size, on_step=lambda m: print(f"  {DIM}{m}{RESET}"))
    _print(rep)
    return 0 if all(r.qc.verdict != Verdict.REJECTED for r in rep.results) else 1


def cmd_batch(a) -> int:
    reps = run_batch(Path(a.root), _cfg(a), on_step=lambda m: print(f"  {DIM}{m}{RESET}"))
    total = sum(r.cost_usd for r in reps)
    ok = sum(1 for r in reps if r.results and all(x.qc.verdict == Verdict.APPROVED for x in r.results))
    human = sum(1 for r in reps if r.needs_human)
    for r in reps:
        _print(r)
    n = len(reps) or 1
    print(f"\n  {'=' * 62}")
    print(f"  {n} tapis | {ok} entierement approuves | {human} a relire")
    print(f"  cout total ${total:.2f} | moyenne ${total / n:.3f} par tapis")
    print(f"  taux d'automatisation complete : {ok / n:.0%}")
    return 0 if human < n else 1


def cmd_dna(a) -> int:
    from babarug.providers.registry import get_vision

    size = tuple(float(x) for x in a.size.split("x")) if a.size else None
    dna = get_vision(a.vision).analyze_rug(a.product_id, load_photos(Path(a.photos)), size)
    out = Path(a.out) / a.product_id / "rug_dna.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dna.model_dump_json(indent=2))
    print(json.dumps(json.loads(dna.model_dump_json()), ensure_ascii=False, indent=2))
    print(f"\n  -> {out}")
    if dna.needs_human_check:
        print(f"  {YELLOW}>> DNA incertain : a verifier avant generation{RESET}")
    return 0


def cmd_scenes(a) -> int:
    """Inventaire et validation de la bibliotheque de decors."""
    from babarug.scenes import SceneLibrary

    lib = SceneLibrary(a.folder)
    print(f"\n  Bibliotheque : {a.folder}\n")
    if not len(lib) and not lib.problems:
        print(f"    {YELLOW}vide{RESET}. Deposez des photos de pieces VIDES dans ce dossier,")
        print(f"    puis calibrez chacune avec tools/calibrer-decor.html\n")
        return 1
    for sc in lib.scenes:
        print(f"    {GREEN}ok{RESET}   {sc.scene_id:32} {sc.interior_style:24} "
              f"{sc.room_type:16} {sc.floor_width_m:.1f}x{sc.floor_depth_m:.1f} m")
    for sid, pbs in lib.problems.items():
        print(f"    {RED}ko{RESET}   {sid}")
        for pb in pbs:
            print(f"           {DIM}{pb}{RESET}")

    print(f"\n    {len(lib)} decor(s) utilisable(s), {len(lib.problems)} ecarte(s)")
    if lib.scenes:
        print(f"    styles : {', '.join(sorted(lib.styles()))}")
        print(f"    pieces : {', '.join(sorted(lib.rooms()))}")
        manque = [r for r in ('salon', 'salle_a_manger', 'chambre') if r not in lib.rooms()]
        if manque:
            print(f"    {YELLOW}!{RESET}  aucune piece de type : {', '.join(manque)}")
        if len(lib) < 6:
            print(f"    {YELLOW}!{RESET}  moins de 6 decors : les tapis se retrouveront souvent "
                  f"dans les memes pieces")
    print()
    return 0 if lib.scenes else 1


def cmd_doctor(a) -> int:
    from babarug.providers.registry import available

    print("\n  Baba Rug Visual Studio -- diagnostic\n")
    ok = True
    for mod in ("cv2", "numpy", "skimage", "PIL", "pydantic", "anthropic"):
        try:
            __import__(mod)
            print(f"    {GREEN}ok{RESET}   {mod}")
        except ImportError:
            print(f"    {RED}absent{RESET} {mod}")
            ok = False
    print()
    for label, var in (("vision (Claude)", "ANTHROPIC_API_KEY"),
                       ("generation (Gemini)", "GEMINI_API_KEY"),
                       ("generation (FLUX)", "BFL_API_KEY"),
                       ("detourage distant", "BABARUG_MATTING_ENDPOINT")):
        if os.environ.get(var):
            print(f"    {GREEN}ok{RESET}   {label}  ({var})")
        else:
            print(f"    {YELLOW}--{RESET}   {label}  ({var} absent)")
    from babarug.scenes import SceneLibrary

    lib = SceneLibrary("scenes")
    if len(lib):
        print(f"    {GREEN}ok{RESET}   bibliotheque de decors : {len(lib)} piece(s), aucun cout")
    else:
        print(f"    {YELLOW}--{RESET}   bibliotheque de decors vide (dossier scenes/)")

    print(f"\n    providers : {available()}")
    print(f"\n    Sans aucune cle : --generation library (decors reels, cout nul)")
    print(f"    {DIM}ou --generation offline pour un decor schematique de test.{RESET}\n")
    return 0 if ok else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser("babarug", description="Baba Rug Visual Studio")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--out", default="data")
        sp.add_argument("--vision", default=None)
        sp.add_argument("--generation", default=None,
                        help="library (decors reels, gratuit) | gemini | flux | offline")
        sp.add_argument("--scenes", default="scenes", help="dossier de la bibliotheque de decors")
        sp.add_argument("--segmentation", default=None, help="local | remote")
        sp.add_argument("--style", nargs="*", default=None)
        sp.add_argument("--room", nargs="*", default=None)
        sp.add_argument("--width", type=int, default=1600)
        sp.add_argument("--height", type=int, default=900)
        sp.add_argument("--max-kb", type=int, default=400)
        sp.add_argument("--max-iterations", type=int, default=3)
        sp.add_argument("--filename", default="{pid}-{room}-tapis-baba-rug")
        sp.add_argument("--no-review", action="store_true",
                        help="desactive l'avis de scene (moins cher, moins sur)")

    r = sub.add_parser("run", help="un tapis")
    r.add_argument("product_id")
    r.add_argument("photos")
    r.add_argument("--size", help="dimensions reelles en cm, ex: 200x150")
    common(r)
    r.set_defaults(func=cmd_run)

    b = sub.add_parser("batch", help="un dossier de shooting entier")
    b.add_argument("root")
    b.add_argument("--size", default=None)
    common(b)
    b.set_defaults(func=cmd_batch)

    d = sub.add_parser("dna", help="analyse seule")
    d.add_argument("product_id")
    d.add_argument("photos")
    d.add_argument("--size")
    d.add_argument("--out", default="data")
    d.add_argument("--vision", default=None)
    d.set_defaults(func=cmd_dna)

    sc = sub.add_parser("scenes", help="inventaire de la bibliotheque de decors")
    sc.add_argument("--folder", default="scenes")
    sc.set_defaults(func=cmd_scenes)

    doc = sub.add_parser("doctor", help="verifie l'installation")
    doc.set_defaults(func=cmd_doctor)

    a = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return a.func(a)
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        print(f"\n  {RED}echec{RESET} : {type(e).__name__}: {e}\n", file=sys.stderr)
        if a.verbose:
            raise
        return 2


if __name__ == "__main__":
    sys.exit(main())
