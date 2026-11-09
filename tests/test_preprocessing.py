import numpy as np

from pipeline_utils.preprocessing import scale_to_8bit_range, crop_frame, rotate_frame


def test_bit_depth_inferred_and_explicit_agree():
    for depth in (10, 12, 14):
        full = (1 << depth) - 1
        raw = np.array([[0, full // 2, full]], dtype=np.uint16)
        np.testing.assert_allclose(
            scale_to_8bit_range(raw), scale_to_8bit_range(raw, depth), rtol=1e-6)


def test_scaling_maps_full_scale_to_255():
    raw = np.array([[4095]], dtype=np.uint16)
    assert scale_to_8bit_range(raw, 12)[0, 0] == 255.0


def test_low_counts_survive_scaling():
    """A uint8 cast would collapse these to zero and lose the dynamic range."""
    raw = np.array([[1, 2, 3]], dtype=np.uint16)
    out = scale_to_8bit_range(raw, 12)
    assert out.dtype == np.float32
    assert len(set(out.ravel().tolist())) == 3
    assert np.all(out > 0)


def test_uint8_passes_through_unscaled():
    raw = np.array([[0, 128, 255]], dtype=np.uint8)
    np.testing.assert_array_equal(scale_to_8bit_range(raw), raw.astype(np.float32))


def test_crop_handles_negative_one_as_end():
    frame = np.arange(100).reshape(10, 10)
    np.testing.assert_array_equal(crop_frame(frame, (2, -1, 3, -1)), frame[2:, 3:])


def test_rotation_preserves_shape():
    frame = np.zeros((32, 48), dtype=np.uint8)
    assert rotate_frame(frame, (16, 24), 8.0).shape == (32, 48)
