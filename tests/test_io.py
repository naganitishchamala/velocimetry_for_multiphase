import numpy as np

from pipeline_utils.io import block_average, block_centers


def test_block_average_centres_match_block_centers():
    """
    The value sampled for each block must sit at the coordinate downstream code
    assumes. A mismatch is a silent spatial registration error.
    """
    window = 4
    ramp = np.arange(32, dtype=np.float32)[:, None, None].repeat(8, 1).repeat(2, 2)
    averaged = block_average(ramp, window, window)
    np.testing.assert_allclose(
        averaged[:, 0, 0], block_centers(averaged.shape[0], window), atol=1e-6)


def test_block_average_is_exact_mean_of_its_block():
    rng = np.random.default_rng(1)
    field = rng.normal(size=(12, 16, 2))
    out = block_average(field, 4, 4)
    np.testing.assert_allclose(out[1, 2], field[4:8, 8:12].mean(axis=(0, 1)))


def test_ragged_edges_are_dropped():
    field = np.zeros((13, 17, 2))
    assert block_average(field, 4, 4).shape == (3, 4, 2)
