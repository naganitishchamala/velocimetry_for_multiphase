#!/usr/bin/env python3
"""
Endpoint error (EPE) of RAFT, Farneback and OpenPIV on synthetic image pairs
whose true displacement is known.

Every method runs through the same wrapper and the same preprocessing that
``tracker.py`` applies to real recordings, so the numbers describe the
pipeline as it is used, not a separate evaluation copy of each method.

Input
-----
A pairs file with one test pair per line, whitespace separated::

    path/to/02305_img1.png  path/to/02305_img2.png  path/to/02305_flow.flo

Relative paths resolve against the pairs file's own directory, and lines
starting with ``#`` are ignored.  Alternatively, ``--data_dir`` finds pairs by
filename suffix (see ``--img1_suffix``, ``--img2_suffix``, ``--flow_suffix``).

To score exactly the pairs that were held out during RAFT training, write the
pairs file from the training code's own validation dataset.  RAFT's
``FlowDataset`` subclasses keep the lists this needs::

    ds = ...  # the validation dataset, built as evaluate.py builds it
    with open('test_pairs.txt', 'w') as f:
        for (img1, img2), flo in zip(ds.image_list, ds.flow_list):
            f.write(f'{img1} {img2} {flo}\\n')

Ground truth
------------
``.flo`` files are read as stored: u positive to the right, v positive down
the image rows.  That is the convention of the ``.flo`` format, of RAFT and
Farneback, and of the OpenPIV wrapper here (checked on a known shift with
openpiv 0.26).  Nothing is flipped.  The summary reports the mean u and v of
the ground truth and of every prediction, so a sign error shows up directly,
as does an EPE larger than the zero-motion baseline.

Mask
----
One mask per pair, taken from the ground truth alone so every method is
scored on the same pixels: the ground truth must be finite and, unless
``--keep_zero_gt`` is given, not exactly zero in both components.  Exact
zeros mark the zero-filled area outside the CFD domain and the no-slip walls.
``--mask`` (an image, non-zero = scored) and ``--border`` narrow it further.

Pixels inside the mask where a method returns NaN (OpenPIV leaves its border
and rejected vectors empty) are *not* silently dropped.  They are counted in
``coverage``, and ``epe_mean_gaps_as_zero`` scores them as if the method had
predicted no motion.

Output
------
::

    <out_dir>/<method>/<stem>.flo   predictions, NaN where a method gave none
    <out_dir>/per_pair.csv          one row per method and pair
    <out_dir>/summary.csv           per block and pooled ("all")
    <out_dir>/run_info.json         arguments, library versions, weight hashes

EPE statistics are pooled over pixels, as RAFT's own validation reports
them.  ``zero-motion`` is a baseline row whose EPE is the ground-truth
magnitude, which any working method must beat.

Examples
--------
::

    # all three methods, both RAFT models, pipeline preprocessing with CLAHE
    python evaluate_epe.py --pairs test_pairs.txt --out_dir epe_clahe \\
        --methods raft farneback openpiv \\
        --raft_models weights/raft-cloudcav.pth weights/raft-sintel.pth \\
        --use_clahe

    # quick check on the first 5 pairs
    python evaluate_epe.py --pairs test_pairs.txt --out_dir epe_try --limit 5

    # re-score existing predictions (e.g. after changing the mask)
    python evaluate_epe.py --pairs test_pairs.txt --out_dir epe_clahe --score_only
"""

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import OrderedDict

import cv2
import numpy as np

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

FLO_TAG = 202021.25
ZERO_MOTION = 'zero-motion'


# --------------------------------------------------------------------------
# .flo input/output
# --------------------------------------------------------------------------

def read_flo(path):
    """Read a Middlebury .flo file as an (H, W, 2) float32 array."""
    with open(path, 'rb') as f:
        tag = np.fromfile(f, np.float32, count=1)
        if tag.size != 1 or tag[0] != np.float32(FLO_TAG):
            raise ValueError(f"{path} is not a .flo file (bad magic number)")
        w, h = (int(n) for n in np.fromfile(f, np.int32, count=2))
        data = np.fromfile(f, np.float32, count=2 * w * h)
    if data.size != 2 * w * h:
        raise ValueError(f"{path} is truncated")
    return data.reshape(h, w, 2)


def write_flo(path, flow):
    """Write an (H, W, 2) array as .flo.  NaN is kept, as a float32 NaN."""
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"expected (H, W, 2) flow, got {flow.shape}")
    h, w = flow.shape[:2]
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        np.array([FLO_TAG], np.float32).tofile(f)
        np.array([w, h], np.int32).tofile(f)
        flow.tofile(f)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# Test pairs
# --------------------------------------------------------------------------

def _stem_of(img1_path, img1_suffix):
    name = os.path.basename(img1_path)
    if img1_suffix and name.endswith(img1_suffix):
        return name[:-len(img1_suffix)]
    stem = os.path.splitext(name)[0]
    return re.sub(r'[_-]?img_?1$', '', stem) or stem


def _unique_stems(pairs):
    """Prefix the parent folder when two pairs share a file stem."""
    stems = [p['stem'] for p in pairs]
    if len(set(stems)) == len(stems):
        return pairs
    for p in pairs:
        parent = os.path.basename(os.path.dirname(p['img1']))
        p['stem'] = f"{parent}_{p['stem']}"
    stems = [p['stem'] for p in pairs]
    if len(set(stems)) != len(stems):
        dup = sorted({s for s in stems if stems.count(s) > 1})[:5]
        raise ValueError(f"test pairs do not have unique names, e.g. {dup}")
    return pairs


def read_pairs_file(path, img1_suffix='_img1.png'):
    base = os.path.dirname(os.path.abspath(path))
    pairs = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"{path}:{lineno}: expected 'img1 img2 flow', got {line!r}")
            img1, img2, flo = (p if os.path.isabs(p) else os.path.join(base, p) for p in parts)
            pairs.append(dict(img1=img1, img2=img2, flow=flo, stem=_stem_of(img1, img1_suffix)))
    return _unique_stems(pairs)


def discover_pairs(data_dir, img1_suffix, img2_suffix, flow_suffix):
    pairs = []
    for name in sorted(os.listdir(data_dir)):
        if not name.endswith(img1_suffix):
            continue
        stem = name[:-len(img1_suffix)]
        img2 = os.path.join(data_dir, stem + img2_suffix)
        flo = os.path.join(data_dir, stem + flow_suffix)
        if os.path.exists(img2) and os.path.exists(flo):
            pairs.append(dict(img1=os.path.join(data_dir, name), img2=img2, flow=flo, stem=stem))
    return _unique_stems(pairs)


def block_label(stem, block_size):
    """Group pairs by their frame number, e.g. 02305 -> '2000-2999'."""
    if not block_size:
        return 'all'
    digits = re.findall(r'\d+', stem)
    if not digits:
        return 'other'
    k = int(digits[-1]) // block_size
    return f"{k * block_size}-{(k + 1) * block_size - 1}"


# --------------------------------------------------------------------------
# Preprocessing, identical to VideoPairDataset.preprocess for grey frames
# --------------------------------------------------------------------------

def load_frame(path, args):
    """
    Read one synthetic image and preprocess it as data.py does a video frame.

    Returns a float32 (H, W) array on the 0-255 scale.
    """
    from pipeline_utils.preprocessing import apply_clahe, scale_to_8bit_range

    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"could not read image {path}")
    if raw.ndim == 3:
        # The renderer writes grey images; a colour file is reduced to grey
        # so every method sees the same single channel.
        raw = cv2.cvtColor(np.ascontiguousarray(raw[..., :3]), cv2.COLOR_BGR2GRAY)

    frame = scale_to_8bit_range(raw, args.bit_depth)
    if args.use_clahe:
        frame = apply_clahe(frame, args.clahe_clip_limit, args.clahe_tile_size,
                            is_grayscale=True)
    return frame


# --------------------------------------------------------------------------
# Methods
# --------------------------------------------------------------------------

def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def method_specs(args):
    """Ordered {label: description} for the methods this run produces."""
    from tracker import resolve_model_path

    specs = OrderedDict()
    for name in args.methods:
        if name == 'raft':
            for model in args.raft_models:
                path = resolve_model_path(model)
                tag = os.path.splitext(os.path.basename(path))[0]
                label = tag if tag.startswith('raft') else f"raft-{tag}"
                specs[label] = dict(kind='raft', model_path=path)
        else:
            specs[name] = dict(kind=name)
    return specs


def build_model(spec, args):
    common = dict(uncert_window=args.uncert_window, uncert_max_shift=args.uncert_search)
    if spec['kind'] == 'raft':
        from models.raft_wrapper import RAFTOpticalFlow
        return RAFTOpticalFlow(model_path=spec['model_path'], device=args.device,
                               iters=args.iters, **common)
    if spec['kind'] == 'farneback':
        from models.farneback_wrapper import FarnebackOpticalFlow
        return FarnebackOpticalFlow(**common)
    if spec['kind'] == 'openpiv':
        from models.openpiv_wrapper import OpenPIVModel
        return OpenPIVModel(**common)
    raise ValueError(f"unknown method {spec['kind']}")


def _batches(pairs, size, shape_of):
    """Consecutive runs of up to ``size`` pairs that share an image shape."""
    batch, shape = [], None
    for p in pairs:
        s = shape_of(p)
        if batch and (s != shape or len(batch) == size):
            yield batch
            batch = []
        batch.append(p)
        shape = s
    if batch:
        yield batch


def _available_cores():
    if hasattr(os, 'sched_getaffinity'):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 4


def run_method(label, spec, pairs, args):
    import torch
    from pipeline_utils.preprocessing import prepare_bgr_to_tensor

    out = os.path.join(args.out_dir, label)
    os.makedirs(out, exist_ok=True)
    todo = [p for p in pairs
            if args.overwrite or not os.path.exists(os.path.join(out, p['stem'] + '.flo'))]
    if not todo:
        print(f"[{label}] all {len(pairs)} predictions exist, skipping (use --overwrite to redo)")
        return

    model = build_model(spec, args)
    size = args.batch_size
    if spec['kind'] == 'openpiv':
        # OpenPIV parallelises over the pairs in a batch, one per core.
        size = args.piv_batch_size or _available_cores()

    def shape_of(p):
        img = cv2.imread(p['img1'], cv2.IMREAD_UNCHANGED)
        return None if img is None else img.shape[:2]

    done = 0
    for batch in _batches(todo, size, shape_of):
        frames1 = [load_frame(p['img1'], args) for p in batch]
        frames2 = [load_frame(p['img2'], args) for p in batch]
        t1 = torch.stack([prepare_bgr_to_tensor(f, is_grayscale=True) for f in frames1])
        t2 = torch.stack([prepare_bgr_to_tensor(f, is_grayscale=True) for f in frames2])
        flow, _ = model.predict_batch(t1, t2)
        flow = np.asarray(flow)
        for p, f in zip(batch, flow):
            write_flo(os.path.join(out, p['stem'] + '.flo'), f)
        done += len(batch)
        print(f"[{label}] {done}/{len(todo)}", flush=True)

    del model
    if spec['kind'] == 'raft' and torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def gt_mask(gt, keep_zero_gt=False, border=0, extra=None):
    """Pixels to score, from the ground truth only (shared by every method)."""
    mask = np.isfinite(gt).all(axis=-1)
    if not keep_zero_gt:
        mask &= ~(gt == 0).all(axis=-1)
    if border > 0:
        edge = np.zeros_like(mask)
        edge[border:-border, border:-border] = True
        mask &= edge
    if extra is not None:
        if extra.shape != mask.shape:
            raise ValueError(f"--mask is {extra.shape}, ground truth is {mask.shape}")
        mask &= extra
    return mask


def score_pair(pred, gt, mask):
    """
    Per-pixel EPE inside ``mask``.

    ``epe`` holds the EPE at covered pixels (where the prediction is finite);
    the rest are sums and counts, so a whole test set fits in memory.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"prediction {pred.shape} and ground truth {gt.shape} differ")
    epe = np.hypot(pred[..., 0] - gt[..., 0], pred[..., 1] - gt[..., 1])
    covered = mask & np.isfinite(epe)
    gaps = mask & ~np.isfinite(epe)
    gt_mag = np.hypot(gt[..., 0], gt[..., 1]).astype(np.float64)
    return dict(
        epe=epe[covered].astype(np.float32),
        n_gaps=int(gaps.sum()),
        mag_covered_sum=float(gt_mag[covered].sum()),
        mag_gaps_sum=float(gt_mag[gaps].sum()),
        pred_uv=pred[covered].astype(np.float64).sum(axis=0),
        gt_uv=gt[covered].astype(np.float64).sum(axis=0),
    )


class Accumulator:
    """Pixel-pooled statistics for one method over one group of pairs."""

    def __init__(self):
        self.n_pairs = 0
        self.epe = []
        self.n_gaps = 0
        self.mag_covered_sum = 0.0
        self.mag_gaps_sum = 0.0
        self.pred_uv = np.zeros(2)
        self.gt_uv = np.zeros(2)

    def add(self, s):
        self.n_pairs += 1
        self.epe.append(s['epe'])
        self.n_gaps += s['n_gaps']
        self.mag_covered_sum += s['mag_covered_sum']
        self.mag_gaps_sum += s['mag_gaps_sum']
        self.pred_uv += s['pred_uv']
        self.gt_uv += s['gt_uv']

    def row(self):
        epe = np.concatenate(self.epe) if self.epe else np.zeros(0, np.float32)
        n_cov = epe.size
        n_px = n_cov + self.n_gaps
        nan = float('nan')

        def pct(q):
            return float(np.percentile(epe, q)) if n_cov else nan

        epe_sum = float(epe.astype(np.float64).sum())
        zero_sum = self.mag_covered_sum
        return OrderedDict(
            n_pairs=self.n_pairs,
            n_px=n_px,
            coverage=n_cov / n_px if n_px else nan,
            epe_mean=epe_sum / n_cov if n_cov else nan,
            epe_median=pct(50),
            epe_p95=pct(95),
            epe_mean_gaps_as_zero=(epe_sum + self.mag_gaps_sum) / n_px if n_px else nan,
            # EPE relative to predicting no motion on the same pixels; above 1
            # means worse than doing nothing, usually a sign or flip error.
            epe_over_zero_motion=epe_sum / zero_sum if zero_sum > 0 else nan,
            gt_mag_mean=zero_sum / n_cov if n_cov else nan,
            gt_u_mean=self.gt_uv[0] / n_cov if n_cov else nan,
            gt_v_mean=self.gt_uv[1] / n_cov if n_cov else nan,
            pred_u_mean=self.pred_uv[0] / n_cov if n_cov else nan,
            pred_v_mean=self.pred_uv[1] / n_cov if n_cov else nan,
        )


def find_prediction(out_dir, label, stem):
    for ext in ('.flo', '.npy'):
        path = os.path.join(out_dir, label, stem + ext)
        if os.path.exists(path):
            return path
    return None


def load_prediction(path):
    if path.endswith('.npy'):
        flow = np.load(path).astype(np.float32)
        if flow.ndim == 3 and flow.shape[0] == 2:
            flow = np.moveaxis(flow, 0, -1)
        return flow
    return read_flo(path)


def score(pairs, labels, args):
    """
    Score every method on every pair.  Methods are scored one at a time so
    only one method's per-pixel errors are held in memory.
    """
    extra = None
    if args.mask:
        extra = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if extra is None:
            raise FileNotFoundError(args.mask)
        extra = extra > 0

    summary, per_pair = [], []
    for label in [ZERO_MOTION] + list(labels):
        groups = OrderedDict()           # block -> Accumulator
        missing = 0
        for p in pairs:
            if label == ZERO_MOTION:
                pred = None
            else:
                path = find_prediction(args.out_dir, label, p['stem'])
                if path is None:
                    missing += 1
                    continue
                pred = load_prediction(path)

            gt = read_flo(p['flow'])
            if pred is None:
                pred = np.zeros_like(gt)
            mask = gt_mask(gt, args.keep_zero_gt, args.border, extra)
            block = block_label(p['stem'], args.block_size)

            s = score_pair(pred, gt, mask)
            for key in dict.fromkeys((block, 'all')):   # once each, even if block is 'all'
                groups.setdefault(key, Accumulator()).add(s)
            n_cov = s['epe'].size
            n_px = n_cov + s['n_gaps']
            nan = float('nan')
            per_pair.append(OrderedDict(
                method=label, stem=p['stem'], block=block, n_px=n_px,
                coverage=n_cov / n_px if n_px else nan,
                epe_mean=float(s['epe'].mean()) if n_cov else nan,
                epe_median=float(np.median(s['epe'])) if n_cov else nan,
                gt_mag_mean=s['mag_covered_sum'] / n_cov if n_cov else nan,
            ))

        if missing:
            print(f"warning: {label} has no prediction for {missing} of {len(pairs)} pairs; "
                  "its rows cover fewer pairs than zero-motion", file=sys.stderr)

        blocks = sorted((b for b in groups if b != 'all'), key=_block_order) + ['all']
        for block in blocks:
            if block in groups:
                summary.append(OrderedDict(method=label, block=block, **groups[block].row()))
    return summary, per_pair


def _block_order(label):
    start = label.split('-')[0]
    return (0, int(start), label) if start.isdigit() else (1, 0, label)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def print_summary(rows):
    cols = ['method', 'block', 'n_pairs', 'coverage', 'epe_mean', 'epe_median', 'epe_p95',
            'epe_mean_gaps_as_zero', 'epe_over_zero_motion', 'gt_v_mean', 'pred_v_mean']
    table = [[str(r[c]) if not isinstance(r[c], float) else f"{r[c]:.3f}" for c in cols]
             for r in rows]
    widths = [max(len(c), *(len(t[i]) for t in table)) for i, c in enumerate(cols)]
    print('  '.join(c.ljust(w) for c, w in zip(cols, widths)))
    for t in table:
        print('  '.join(v.ljust(w) for v, w in zip(t, widths)))


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------

def _version(module):
    try:
        from importlib.metadata import version
        return version(module)
    except Exception:
        return None


def run_info(args, pairs, specs):
    try:
        out = subprocess.run(['git', 'describe', '--always', '--dirty'], cwd=REPO_DIR,
                             capture_output=True, text=True, timeout=5)
        code_version = out.stdout.strip() if out.returncode == 0 else 'unknown'
    except Exception:
        code_version = 'unknown'
    info = OrderedDict(
        code_version=code_version,
        n_pairs=len(pairs),
        pairs_source=os.path.abspath(args.pairs) if args.pairs else os.path.abspath(args.data_dir),
        args=vars(args),
        versions={m: _version(m) for m in ('numpy', 'opencv-python', 'opencv-python-headless',
                                           'torch', 'openpiv', 'scipy')},
        methods=OrderedDict(),
    )
    for label, spec in specs.items():
        entry = dict(spec)
        if spec.get('model_path') and os.path.exists(spec['model_path']):
            entry['sha256'] = file_sha256(spec['model_path'])
        info['methods'][label] = entry
    return info


# --------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Endpoint error of the pipeline's methods on synthetic pairs with known flow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    src = p.add_argument_group('test pairs')
    src.add_argument('--pairs', help="Text file: img1 img2 flow.flo per line")
    src.add_argument('--data_dir', help="Find pairs in this folder by suffix instead")
    src.add_argument('--img1_suffix', default='_img1.png')
    src.add_argument('--img2_suffix', default='_img2.png')
    src.add_argument('--flow_suffix', default='_flow.flo')
    src.add_argument('--limit', type=int, default=0, help="Use only the first N pairs (0 = all)")

    run = p.add_argument_group('methods')
    run.add_argument('--out_dir', required=True)
    run.add_argument('--methods', nargs='+', default=['raft', 'farneback', 'openpiv'],
                     choices=['raft', 'farneback', 'openpiv'])
    run.add_argument('--raft_models', nargs='+', default=['weights/raft-cloudcav.pth'],
                     help="One RAFT run per checkpoint, labelled by file name")
    run.add_argument('--iters', type=int, default=20, help="RAFT iterations, as in tracker.py")
    run.add_argument('--device', default=None, help="Default: cuda if available, else cpu")
    run.add_argument('--batch_size', type=int, default=4, help="Pairs per batch (RAFT, Farneback)")
    run.add_argument('--piv_batch_size', type=int, default=0,
                     help="Pairs per OpenPIV batch (0 = one per available core)")
    run.add_argument('--overwrite', action='store_true', help="Recompute existing predictions")
    run.add_argument('--score_only', action='store_true',
                     help="Score the predictions already in --out_dir without running methods")

    pre = p.add_argument_group('preprocessing (same options as tracker.py)')
    pre.add_argument('--use_clahe', action='store_true')
    pre.add_argument('--clahe_clip_limit', type=float, default=2.0)
    pre.add_argument('--clahe_tile_size', type=int, default=8)
    pre.add_argument('--bit_depth', type=int, default=None)
    pre.add_argument('--uncert_window', type=int, default=8)
    pre.add_argument('--uncert_search', type=int, default=None)

    sc = p.add_argument_group('scoring')
    sc.add_argument('--keep_zero_gt', action='store_true',
                    help="Also score pixels whose ground truth is exactly zero")
    sc.add_argument('--border', type=int, default=0, help="Ignore this many pixels at each edge")
    sc.add_argument('--mask', default=None, help="Image, non-zero where pixels are scored")
    sc.add_argument('--block_size', type=int, default=1000,
                    help="Report pairs grouped by frame number in blocks of this size (0 = off)")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if bool(args.pairs) == bool(args.data_dir):
        parser.error("give exactly one of --pairs or --data_dir")

    if args.pairs:
        pairs = read_pairs_file(args.pairs, args.img1_suffix)
    else:
        pairs = discover_pairs(args.data_dir, args.img1_suffix, args.img2_suffix,
                               args.flow_suffix)
    if args.limit:
        pairs = pairs[:args.limit]
    if not pairs:
        parser.error("no test pairs found")
    missing = [p[k] for p in pairs for k in ('img1', 'img2', 'flow') if not os.path.exists(p[k])]
    if missing:
        parser.error(f"{len(missing)} listed files do not exist, e.g. {missing[0]}")

    os.makedirs(args.out_dir, exist_ok=True)

    if args.score_only:
        labels = [d for d in sorted(os.listdir(args.out_dir))
                  if os.path.isdir(os.path.join(args.out_dir, d)) and d != ZERO_MOTION]
        specs = OrderedDict((label, dict(kind='existing')) for label in labels)
    else:
        if args.device is None:
            import torch
            args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        specs = method_specs(args)
        for label, spec in specs.items():
            run_method(label, spec, pairs, args)
        labels = list(specs)

    # A re-score keeps the original run's record (versions, weight hashes).
    info_name = 'score_info.json' if args.score_only else 'run_info.json'
    with open(os.path.join(args.out_dir, info_name), 'w') as f:
        json.dump(run_info(args, pairs, specs), f, indent=2, default=str)

    summary, per_pair = score(pairs, labels, args)
    write_csv(os.path.join(args.out_dir, 'summary.csv'), summary)
    write_csv(os.path.join(args.out_dir, 'per_pair.csv'), per_pair)
    print_summary(summary)
    print(f"\nWrote {os.path.join(args.out_dir, 'summary.csv')} and per_pair.csv")
    return 0


if __name__ == '__main__':
    sys.exit(main())
