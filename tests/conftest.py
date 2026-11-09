import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def synthetic_video(tmp_path):
    """A short lossless video of a texture translating by a known 1 px/frame."""
    path = str(tmp_path / "synthetic.avi")
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (64, 96), dtype=np.uint8)

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'FFV1'), 30, (96, 64))
    assert writer.isOpened(), "FFV1 encoder unavailable"
    for k in range(12):
        frame = np.roll(base, k, axis=1)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    return path
