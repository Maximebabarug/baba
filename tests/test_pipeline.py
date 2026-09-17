"""Styles, export, et chaine complete hors ligne (ni cle API ni cout)."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from babarug.models import Verdict
from babarug.pipeline.export import build_filename, export_jpeg, slugify
from babarug.run import RunConfig, run_batch, run_product
from babarug.styles import DISCOURAGED, build_brief, surprise_me


# ---------------------------------------------------------------- styles
def test_surprends_moi_donne_deux_scenes_franchement_differentes():
    (s1, r1), (s2, r2) = surprise_me("BABA-RUG-0001", 2)
    assert s1 != s2 and r1 != r2
    assert (s1, r1) not in DISCOURAGED and (s2, r2) not in DISCOURAGED


def test_les_styles_sont_reproductibles():
    assert surprise_me("BABA-RUG-0007") == surprise_me("BABA-RUG-0007")


def test_des_tapis_differents_donnent_des_scenes_differentes():
    combos = {tuple(surprise_me(f"BABA-RUG-{i:04d}")) for i in range(1, 25)}
    assert len(combos) > 6, "le moteur de variation produit des scenes trop repetitives"


def test_le_prompt_de_scene_ne_demande_jamais_de_tapis():
    """Regle d'architecture : le generateur ne dessine pas le produit."""
    for pid in ("A", "B", "C"):
        for style, room in surprise_me(pid, 2):
            p = build_brief(pid, style, room, "A").prompt().lower()
            assert "bare" in p and "no rug" in p
            # aucune injonction a dessiner un tapis
            assert "with a rug" not in p and "place a rug" not in p


# ---------------------------------------------------------------- export
def test_slug_et_nom_de_fichier():
    assert slugify("Salle à manger") == "salle-a-manger"
    assert build_filename("quelle-taille-tapis-{room}-baba-rug", room="Salon") == \
        "quelle-taille-tapis-salon-baba-rug.jpg"


def test_export_respecte_le_format_et_le_poids(tmp_path):
    img = np.random.default_rng(0).integers(0, 255, (1200, 1200, 3), dtype=np.uint8)
    info = export_jpeg(img, tmp_path / "o.jpg", 1600, 900, max_kb=250)
    out = cv2.imread(str(tmp_path / "o.jpg"))
    assert out.shape[:2] == (900, 1600)
    assert info["quality"] >= 68


# ---------------------------------------------------------------- bout en bout
@pytest.fixture
def produit(tmp_path, rug_photo):
    photo, _ = rug_photo
    src = tmp_path / "photos" / "BABA-TEST-9"
    src.mkdir(parents=True)
    cv2.imwrite(str(src / "01_face.jpg"), photo)
    out = tmp_path / "out" / "BABA-TEST-9"
    out.mkdir(parents=True)
    (out / "rug_dna.json").write_text(json.dumps({
        "product_id": "BABA-TEST-9", "n_photos": 1, "shape": "rectangular",
        "aspect_ratio": 1400 / 900, "aspect_ratio_confidence": 1.0, "orientation": "portrait",
        "real_size_cm": [210, 135],
        "dominant_colors": [{"name": "rouge", "hex": "#822A26", "coverage": 0.6}],
        "secondary_colors": [], "pattern_description": "semis", "border_description": "triple",
        "central_motif": "medaillon", "repetition": "6x4", "fringe": "ivoire",
        "texture": "laine", "material_guess": "laine", "pile_height": "8mm",
        "wear": "legere", "irregularities": ["abrash"], "unique_features": ["medaillon"],
        "photos": [{"filename": "01_face.jpg", "view": "full_flat", "shows_full_rug": True,
                    "rug_coverage": 0.6, "perspective_severity": 0.15, "sharpness": 0.9,
                    "color_cast": "neutre", "usable_as_plate": True, "notes": ""}],
        "plate_photo": "01_face.jpg", "photographic_artifacts": [], "analysis_notes": "",
    }))
    return tmp_path, src


def test_chaine_complete_hors_ligne(produit):
    """8 photos -> DNA -> plate -> 2 lifestyles -> QC -> export, sans API."""
    tmp, src = produit
    cfg = RunConfig(out_root=tmp / "out", generation="offline", segmentation="local")
    cfg.settings.review_with_model = False
    rep = run_product("BABA-TEST-9", src, cfg)

    assert len(rep.results) == 2, "deux lifestyles doivent etre produits"
    assert rep.results[0].brief.slug != rep.results[1].brief.slug, "les scenes doivent differer"
    assert rep.cost_usd == 0.0
    for r in rep.results:
        assert r.qc.verdict != Verdict.REJECTED, r.qc.failures
        assert r.qc.fidelity.delta_e_mean < 3.0
        assert r.qc.fidelity.visible_fraction > 0.98
        assert Path(r.export_path).exists()
        assert cv2.imread(r.export_path).shape[:2] == (900, 1600)
    assert (tmp / "out" / "BABA-TEST-9" / "report.json").exists()


def test_un_tapis_en_echec_n_interrompt_pas_le_lot(produit):
    """Exigence de production : un lot de shooting doit aller au bout."""
    tmp, src = produit
    (src.parent / "BABA-CASSE").mkdir()  # dossier vide -> echec attendu
    cfg = RunConfig(out_root=tmp / "out", generation="offline", segmentation="local")
    cfg.settings.review_with_model = False
    reps = run_batch(src.parent, cfg)
    assert len(reps) == 2
    casse = next(r for r in reps if r.product_id == "BABA-CASSE")
    assert casse.needs_human and any("ECHEC" in w for w in casse.warnings)
    bon = next(r for r in reps if r.product_id == "BABA-TEST-9")
    assert len(bon.results) == 2
