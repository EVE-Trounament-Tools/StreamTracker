from pathlib import Path

import cv2
import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
# Fixtures store only the HUD strip (rows 740-1079) of each 1920x1080 frame.
STRIP_TOP = 740


@pytest.fixture(scope="session")
def frame():
    cache = {}

    def load(offset: int):
        if offset not in cache:
            strip = cv2.imread(str(FIXTURES / f"f_{offset:04d}.png"))
            top = np.zeros((STRIP_TOP, strip.shape[1], 3), np.uint8)
            cache[offset] = np.vstack([top, strip])
        return cache[offset].copy()

    return load
