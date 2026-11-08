"""EPE evaluation: file format, mask, gap accounting and sign conventions."""

import os

import cv2
import numpy as np
import pytest

import evaluate_epe as ev


def _shear_gt(h=40, w=60):
    y = np.arange(h, dtype=np.float32)[:, None]
    u = np.broadcast_to(1.0 + 0.1 * y, (h, w))
    v = np.full((h, w), 0.5, np.float32)
    return np.stack([u, v], axis=-1).astype(np.float32)


def test_flo_round_trip_keeps_nan(tmp_path):
    flow = _shear_gt()
    flow[3, 4] = np.nan
    path = str(tmp_path / 'a.flo')
    ev.write_flo(path, flow)
    back = ev.read_flo(path)
    assert back.shape == flow.shape
    assert np.isnan(back[3, 4]).all()
    np.testing.assert_array_equal(np.nan_to_num(back), np.nan_to_num(flow))


def test_mask_drops_zero_filled_ground_truth_and_border():
    gt = _shear_gt()
    gt[:5] = 0.0                      # outside the CFD domain
    mask = ev.gt_mask(gt, border=2)
    assert not mask[:5].any()
    assert not mask[:, :2].any() and not mask[:, -2:].any()
    assert mask[10, 10]
    assert ev.gt_mask(gt, keep_zero_gt=True)[:5].all()


def test_gaps_are_counted_not_silently_dropped():
    gt = _shear_gt()
    pred = gt.copy()
    pred[:, :30] = np.nan             # the method covers only half the field
    mask = ev.gt_mask(gt)
    acc = ev.Accumulator()
    acc.add(ev.score_pair(pred, gt, mask))
    row = acc.row()
    assert row['coverage'] == pytest.approx(0.5)
    assert row['epe_mean'] == pytest.approx(0.0)
    # Gaps scored as zero motion cost |gt| each.
    mag = np.hypot(gt[..., 0], gt[..., 1])
    assert row['epe_mean_gaps_as_zero'] == pytest.approx(mag[:, :30].sum() / mag.size, rel=1e-5)


def test_flipped_ground_truth_shows_in_the_summary():
    """The bug in the old Farneback/PIV scripts: flipping rows and negating v."""
    gt = _shear_gt()
    flipped = gt[::-1].copy()
    flipped[..., 1] *= -1
    mask = ev.gt_mask(gt)

    good = ev.Accumulator()
    good.add(ev.score_pair(gt, gt, mask))
    bad = ev.Accumulator()
    bad.add(ev.score_pair(flipped, gt, mask))

    assert good.row()['epe_over_zero_motion'] == pytest.approx(0.0)
    assert bad.row()['epe_mean'] >= 1.0      # |2 v| everywhere, plus the u mismatch
    # The mean v columns expose the sign error directly.
    assert bad.row()['pred_v_mean'] == pytest.approx(-bad.row()['gt_v_mean'])


@pytest.mark.parametrize('stem, size, label', [
    ('02305', 1000, '2000-2999'),
    ('00017', 1000, '0-999'),
    ('scene_12', 10, '10-19'),
    ('02305', 0, 'all'),
    ('frame', 1000, 'other'),
])
def test_block_label(stem, size, label):
    assert ev.block_label(stem, size) == label


def test_pairs_file_resolves_relative_paths_and_skips_comments(tmp_path):
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'pairs.txt').write_text(
        '# held-out pairs\n\nsub/00007_img1.png sub/00007_img2.png sub/00007_flow.flo\n')
    pairs = ev.read_pairs_file(str(tmp_path / 'pairs.txt'))
    assert len(pairs) == 1
    assert pairs[0]['stem'] == '00007'
    assert pairs[0]['img1'] == os.path.join(str(tmp_path), 'sub', '00007_img1.png')


def _write_shifted_pair(folder, stem, du=2, dv=1, h=208, w=320):
    """Smoothed speckle moved du px right and dv px down (rows increasing)."""
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur(rng.uniform(0, 255, (h, w)).astype(np.float32), (0, 0), 1.0)
    base = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX)
    img1 = base.round().astype(np.uint8)
    img2 = np.roll(img1, (dv, du), axis=(0, 1))
    cv2.imwrite(os.path.join(folder, f'{stem}_img1.png'), img1)
    cv2.imwrite(os.path.join(folder, f'{stem}_img2.png'), img2)
    gt = np.zeros((h, w, 2), np.float32)
    gt[..., 0], gt[..., 1] = du, dv
    ev.write_flo(os.path.join(folder, f'{stem}_flow.flo'), gt)


@pytest.mark.parametrize('method', ['farneback', 'openpiv'])
def test_methods_report_v_positive_down_like_the_ground_truth(tmp_path, method):
    """Guards the sign convention, which has differed between OpenPIV versions."""
    if method == 'openpiv':
        pytest.importorskip('openpiv')
    data = tmp_path / 'data'
    data.mkdir()
    _write_shifted_pair(str(data), '00001')
    out = tmp_path / 'out'
    assert ev.main(['--data_dir', str(data), '--out_dir', str(out), '--methods', method,
                    '--border', '16', '--device', 'cpu']) == 0

    pred = ev.read_flo(str(out / method / '00001.flo'))
    inner = pred[16:-16, 16:-16]
    assert np.nanmedian(inner[..., 0]) == pytest.approx(2.0, abs=0.25)
    assert np.nanmedian(inner[..., 1]) == pytest.approx(1.0, abs=0.25)
    assert (out / 'summary.csv').exists() and (out / 'run_info.json').exists()
