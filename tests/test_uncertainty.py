"""
The estimator measures a *signed residual displacement error*, not a standard
deviation. These tests pin that meaning down, because every downstream formula
depends on it.
"""

import numpy as np
import pytest
import torch

from pipeline_utils.uncertainty import compute_uncertainty_batch


@pytest.fixture
def texture():
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(0)
    img = gaussian_filter(rng.random((128, 192)).astype(np.float32), 1.2)
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    return img


def _run(img1, img2, flow_x):
    t1 = torch.from_numpy(img1)[None, None]
    t2 = torch.from_numpy(img2)[None, None]
    flow = torch.zeros(1, 2, *img1.shape)
    flow[:, 0] = flow_x
    return compute_uncertainty_batch(t1, t2, flow)


def test_perfect_prediction_has_near_zero_residual(texture):
    result = _run(texture, texture, 0.0)
    valid = result.valid
    assert np.abs(result.residual[..., 0][valid]).mean() < 0.05


def test_residual_recovers_a_known_error(texture):
    """A displacement the tracker got wrong by 1 px must read as 1 px."""
    shifted = np.roll(texture, 3, axis=1)
    result = _run(texture, shifted, 2.0)          # true 3, predicted 2
    valid = result.valid
    assert np.abs(result.residual[..., 0][valid]).mean() == pytest.approx(1.0, abs=0.05)


def test_residual_sign_distinguishes_over_from_under_prediction(texture):
    shifted = np.roll(texture, 3, axis=1)
    under = _run(texture, shifted, 2.0)
    over = _run(texture, shifted, 4.0)
    mean_under = under.residual[..., 0][under.valid].mean()
    mean_over = over.residual[..., 0][over.valid].mean()
    assert np.sign(mean_under) == -np.sign(mean_over)
    assert abs(mean_under) == pytest.approx(abs(mean_over), abs=0.1)


def test_sign_convention_is_flow_minus_residual(texture):
    """Documented contract: flow_corrected = flow - residual."""
    shifted = np.roll(texture, 3, axis=1)
    result = _run(texture, shifted, 2.0)
    corrected = 2.0 - result.residual[..., 0][result.valid].mean()
    assert corrected == pytest.approx(3.0, abs=0.05)


def test_errors_beyond_the_search_window_are_flagged_not_clamped(texture):
    """
    A residual larger than the search half-width cannot be measured. It must be
    reported as invalid rather than silently reported as a smaller number.
    """
    shifted = np.roll(texture, 3, axis=1)
    ok = _run(texture, shifted, 2.0)
    beyond = _run(texture, shifted, -3.0)         # true error 6 px, search is +/-4
    assert beyond.valid.mean() < 0.5 * ok.valid.mean()


def test_sigma_is_the_magnitude_of_the_residual(texture):
    result = _run(texture, np.roll(texture, 3, axis=1), 2.0)
    np.testing.assert_allclose(result.sigma, np.abs(result.residual))


def test_frame_border_is_marked_invalid(texture):
    result = _run(texture, texture, 0.0)
    assert not result.valid[0, 0, 0]
    assert result.valid[0, 64, 96]


def test_decorrelated_input_is_mostly_invalid():
    """Pure noise carries no displacement information; it must not look confident."""
    rng = np.random.default_rng(1)
    a = rng.random((128, 192)).astype(np.float32) * 255
    b = rng.random((128, 192)).astype(np.float32) * 255
    result = _run(a, b, 0.0)
    assert result.valid.mean() < 0.35
