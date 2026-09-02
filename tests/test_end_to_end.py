"""Full pipeline on a synthetic recording with a known displacement."""

import os
import subprocess
import sys

import h5py
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_tracker(video, extra=()):
    cmd = [sys.executable, os.path.join(ROOT, 'tracker.py'),
           '--path', video, '--method', 'farneback',
           '--imgScale', '0.05', '--fpsCam', '1000',
           '--win_h', '4', '--win_w', '4', *extra]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    return os.path.splitext(video)[0] + '_avi_fb_c1.h5'


def test_calibration_arguments_are_required(synthetic_video):
    """A forgotten scale must fail loudly, not silently scale every result."""
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, 'tracker.py'), '--path', synthetic_video],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode != 0
    assert '--imgScale' in proc.stderr


def test_tracker_writes_expected_datasets_and_metadata(synthetic_video):
    out = _run_tracker(synthetic_video)
    with h5py.File(out, 'r') as f:
        for name in ('velocity', 'uncertainty', 'residual', 'valid_fraction'):
            assert name in f, f"missing dataset: {name}"
        assert f['velocity'].shape == (11, 16, 24, 2)

        attrs = f['velocity'].attrs
        for key in ('mm_per_px', 'fps_capture', 'method', 'model',
                    'window_width', 'window_height', 'n_frames', 'code_version'):
            assert key in attrs, f"missing provenance attribute: {key}"
        assert attrs['mm_per_px'] == pytest.approx(0.05)
        assert attrs['method'] == 'farneback'

        # Attributes are mirrored so any dataset read alone is self-describing.
        assert f['uncertainty'].attrs['mm_per_px'] == pytest.approx(0.05)


def test_tracked_flow_recovers_the_known_displacement(synthetic_video):
    """The fixture translates by exactly 1 px/frame in +x."""
    out = _run_tracker(synthetic_video)
    with h5py.File(out, 'r') as f:
        u = f['velocity'][:, :, :, 0]
        valid = f['valid_fraction'][:] > 0.5
    assert np.median(u[valid]) == pytest.approx(1.0, abs=0.3)


def test_profiles_carry_calibration_forward(synthetic_video):
    from postprocess.velocity import extract_line_profiles
    out = _run_tracker(synthetic_video)
    _, lines_path = extract_line_profiles(out, [0.0], estimate_timescale=False)

    with h5py.File(lines_path, 'r') as f:
        assert f.attrs['mm_per_px'] == pytest.approx(0.05)
        assert f.attrs['fps_capture'] == pytest.approx(1000)
        assert f.attrs['method'] == 'farneback'
        grp = f['x_location_0_0']
        for name in ('mean_velocity', 'mean_velocity_uncertainty',
                     'mean_velocity_bias', 'instantaneous_precision',
                     'reynolds_stresses', 'reynolds_stresses_noise_corrected',
                     'n_samples', 'n_effective'):
            assert name in grp, f"missing profile dataset: {name}"


def test_angle_argument_refuses_rather_than_silently_ignoring(synthetic_video):
    from postprocess.velocity import extract_line_profiles
    out = _run_tracker(synthetic_video)
    with pytest.raises(NotImplementedError):
        extract_line_profiles(out, [0.0], angle_deg=8.0)


def test_shedding_frequency_requires_an_explicit_rate(synthetic_video):
    from postprocess.video import compute_shedding_frequency
    with pytest.raises(ValueError, match='fps is required'):
        compute_shedding_frequency(synthetic_video)
