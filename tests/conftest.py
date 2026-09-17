import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from synth import synth_room, synth_rug_photo  # noqa: E402

from babarug.geometry import FloorPlane  # noqa: E402
from babarug.pipeline.plate import extract_plate  # noqa: E402

ASPECT = 1400 / 900


@pytest.fixture(scope="session")
def rug_photo():
    return synth_rug_photo()


@pytest.fixture(scope="session")
def plate(rug_photo):
    photo, mask = rug_photo
    return extract_plate(photo, mask, aspect_ratio=ASPECT)


@pytest.fixture(scope="session")
def room():
    img, quad = synth_room()
    return img, FloorPlane(quad, width_m=4.2, depth_m=3.4)
