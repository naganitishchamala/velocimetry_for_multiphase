#!/usr/bin/env python3
"""
Batch optical flow / PIV velocimetry over high-speed video.

Reads a .cine or .avi recording, estimates dense displacement fields with the
selected backend, estimates a per-pixel residual uncertainty for each field, and
streams window-averaged results to a chunked HDF5 file.
"""

import argparse
import os
import subprocess
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import VideoPairDataset
from pipeline_utils.io import HDF5Writer


def _code_version():
    """Short git description of the working tree, for provenance."""
    try:
        out = subprocess.run(
            ['git', 'describe', '--always', '--dirty'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return 'unknown'


def resolve_model_path(model):
    """Accept a path, a bare name, or a name without the .pth suffix."""
    candidates = [model,
                  os.path.join('weights', os.path.basename(model)),
                  os.path.join('weights', os.path.basename(model) + '.pth')]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return model


def build_model(args, device):
    common = dict(uncert_window=args.uncert_window,
                  uncert_max_shift=args.uncert_search)

    base_path, ext = os.path.splitext(args.path)
    base_name = f"{base_path}_{ext[1:].lower()}"

    if args.method == 'raft':
        from models.raft_wrapper import RAFTOpticalFlow
        model_path = resolve_model_path(args.model)
        model = RAFTOpticalFlow(
            model_path=model_path, device=device, small=args.small,
            mixed_precision=args.mixed_precision, alternate_corr=args.alternate_corr,
            iters=args.iters, **common)
        tag = os.path.splitext(os.path.basename(model_path))[0]
        return model, f"{base_name}_{tag}.h5", model_path

    if args.method == 'farneback':
        from models.farneback_wrapper import FarnebackOpticalFlow
        return FarnebackOpticalFlow(**common), f"{base_name}_fb_{args.model}.h5", args.model

    if args.method == 'openpiv':
        from models.openpiv_wrapper import OpenPIVModel
        return OpenPIVModel(**common), f"{base_name}_piv_{args.model}.h5", args.model

    raise ValueError(f"Unknown method: {args.method}")


def main(args):
    print("Initializing video dataset...")
    dataset = VideoPairDataset(
        video_path=args.path,
        roi=args.roi,
        use_clahe=args.use_clahe,
        clahe_clip_limit=args.clahe_clip_limit,
        clahe_tile_size=args.clahe_tile_size,
        max_frames=args.frames,
        rotate_angle=args.rotate_angle,
        rotate_center=tuple(args.throat_loc) if args.throat_loc else None,
        bit_depth=args.bit_depth,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Initializing {args.method.upper()} model on {device.upper()}...")
    model, output_path, model_id = build_model(args, device)

    # OpenPIV fans a batch out across a process pool; RAFT batches on the GPU.
    dl_batch_size = 16 if args.method == 'openpiv' else args.batch_size

    dataloader = DataLoader(dataset, batch_size=dl_batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)

    sample_img1, _, _ = dataset[0]
    _, h, w = sample_img1.shape

    metadata = {
        'window_height': args.win_h,
        'window_width': args.win_w,
        'mm_per_px': args.imgScale,
        'fps_capture': args.fpsCam,
        'roi': args.roi,
        'method': args.method,
        'model': model_id,
        'raft_iters': args.iters if args.method == 'raft' else 0,
        'uncert_window': args.uncert_window,
        'uncert_max_shift': args.uncert_search if args.uncert_search else args.uncert_window // 2,
        'use_clahe': int(args.use_clahe),
        'clahe_clip_limit': args.clahe_clip_limit,
        'clahe_tile_size': args.clahe_tile_size,
        'bit_depth': args.bit_depth if args.bit_depth else 0,
        'n_frames': dataset.frame_count,
        'source_video': os.path.basename(args.path),
        'code_version': _code_version(),
    }
    if args.throat_loc is not None:
        metadata['throat_loc_px'] = args.throat_loc
    if args.rotate_angle != 0.0:
        metadata['rotate_angle'] = args.rotate_angle

    print(f"Writing HDF5 output to: {output_path}")
    writer = HDF5Writer(
        output_path=output_path, frame_count=dataset.frame_count, h=h, w=w,
        window_height=args.win_h, window_width=args.win_w, metadata=metadata)

    try:
        for image1_batch, image2_batch, indices in tqdm(dataloader, desc="Processing batches"):
            flow_batch, uncertainty = model.predict_batch(image1_batch, image2_batch)
            writer.write_batch(indices[0].item(), flow_batch, uncertainty)
    finally:
        writer.close()
        dataset.close()

    print(f"Done. Wrote {dataset.frame_count - 1} frame pairs to {output_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Batch optical flow / PIV velocimetry over high-speed video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    core = parser.add_argument_group('input and method')
    core.add_argument('--path', required=True, help="Input video file (.cine or .avi)")
    core.add_argument('--method', default='farneback', choices=['raft', 'farneback', 'openpiv'])
    core.add_argument('--model', default='c1',
                      help="RAFT: weights path or name in weights/. Others: a free-form run ID.")
    core.add_argument('-n', '--frames', type=int, default=-1,
                      help="Process only the first N frames; -1 for all")

    # Calibration cannot be inferred from the video and is wrong by default for
    # anyone but the original author, so both are required.
    calib = parser.add_argument_group('calibration (required)')
    calib.add_argument('--imgScale', type=float, required=True,
                       help="Image scale in mm per pixel")
    calib.add_argument('--fpsCam', type=int, required=True,
                       help="Camera capture rate in Hz")

    grid = parser.add_argument_group('output grid')
    grid.add_argument('--win_h', type=int, default=4, help="Spatial averaging window height (px)")
    grid.add_argument('--win_w', type=int, default=4, help="Spatial averaging window width (px)")
    grid.add_argument('--roi', type=int, nargs=4, default=[0, -1, 0, -1],
                      metavar=('Y1', 'Y2', 'X1', 'X2'), help="Crop before processing; -1 = to end")
    grid.add_argument('--throat_loc', type=int, nargs=2, default=None, metavar=('Y', 'X'),
                      help="Reference point in pixels, saved to metadata and used as rotation centre")
    grid.add_argument('--rotate_angle', type=float, default=0.0,
                      help="Pre-rotate by this many degrees CCW about --throat_loc")

    unc = parser.add_argument_group('uncertainty')
    unc.add_argument('--uncert_window', type=int, default=8,
                     help="NCC correlation window for the uncertainty estimate (px)")
    unc.add_argument('--uncert_search', type=int, default=None,
                     help="Half-width of the residual search (px). Defaults to uncert_window//2. "
                          "Must exceed the expected residual error, which grows with "
                          "per-frame displacement; residuals beyond it are flagged invalid.")

    proc = parser.add_argument_group('processing')
    proc.add_argument('--batch_size', type=int, default=1, help="Frame pairs per batch")
    proc.add_argument('--bit_depth', type=int, default=None,
                      help="Sensor bit depth for raw 16-bit input. Inferred if omitted.")
    proc.add_argument('--use_clahe', action='store_true', help="Apply CLAHE to each frame")
    proc.add_argument('--clahe_clip_limit', type=float, default=2.0)
    proc.add_argument('--clahe_tile_size', type=int, default=8)

    raft = parser.add_argument_group('RAFT options')
    raft.add_argument('--iters', type=int, default=20, help="GRU refinement iterations")
    raft.add_argument('--small', action='store_true')
    raft.add_argument('--mixed_precision', action='store_true')
    raft.add_argument('--alternate_corr', action='store_true')

    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    if args.rotate_angle != 0.0 and args.throat_loc is None:
        parser.error('--throat_loc is required when --rotate_angle is non-zero.')
    if args.imgScale <= 0:
        parser.error('--imgScale must be positive.')
    if args.fpsCam <= 0:
        parser.error('--fpsCam must be positive.')

    sys.exit(main(args))
