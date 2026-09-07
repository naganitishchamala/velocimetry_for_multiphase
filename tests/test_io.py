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


def test_legacy_block_centers_match_the_old_writer():
    """
    Files written before 0.2.0 must be read with the convention they were
    written under, or every profile is misplaced by window // 2 pixels.
    """
    import cv2
    from pipeline_utils.io import legacy_block_centers

    for window in (2, 3, 4, 8):
        height = 8 * window
        pad = window // 2
        ramp = np.arange(height, dtype=np.float32)[:, None].repeat(4, 1)
        kernel = np.ones([window, window], dtype=np.float32) / (window * window)

        padded = np.pad(ramp, ((pad, pad), (pad, pad)), mode='reflect')
        old = cv2.filter2D(padded, -1, kernel)[pad:-pad:window, pad:-pad:window][:, 0]

        # Block 0 is contaminated by the reflection padding and is not a clean
        # average of any source region, so it is excluded.
        np.testing.assert_allclose(
            old[1:], legacy_block_centers(len(old), window)[1:], atol=1e-5)


def test_legacy_and_current_centers_differ_by_half_a_window():
    from pipeline_utils.io import legacy_block_centers

    for window in (2, 4, 8):
        offset = block_centers(5, window) - legacy_block_centers(5, window)
        np.testing.assert_allclose(offset, window // 2)


def test_legacy_detection_keys_off_code_version():
    from pipeline_utils.io import uses_legacy_block_centers

    assert uses_legacy_block_centers({'mm_per_px': 0.05})
    assert not uses_legacy_block_centers({'mm_per_px': 0.05, 'code_version': 'abc123'})
