"""Frame preprocessing: rotation, cropping, contrast enhancement, tensor conversion."""

import cv2
import numpy as np
import torch


def rotate_frame(frame, center, angle_deg):
    """
    Rotate a frame around a centre point.

    Args:
        frame: (H, W) or (H, W, C) array.
        center: (y, x) rotation centre in pixels.
        angle_deg: degrees, positive = counter-clockwise.
    """
    h, w = frame.shape[:2]
    cv2_center = (float(center[1]), float(center[0]))
    matrix = cv2.getRotationMatrix2D(cv2_center, angle_deg, 1.0)
    return cv2.warpAffine(frame, matrix, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def crop_frame(frame, roi):
    """Crop to ``(y_start, y_end, x_start, x_end)``; -1 means 'to the end'."""
    y_start, y_end, x_start, x_end = roi
    y_slice = slice(y_start, y_end if y_end != -1 else None)
    x_slice = slice(x_start, x_end if x_end != -1 else None)
    return frame[y_slice, x_slice]


def scale_to_8bit_range(frame, bit_depth=None):
    """
    Scale raw sensor counts to the 0-255 range **as float32**, without quantising.

    High-speed cameras store 10-, 12- or 14-bit samples in a uint16 container.
    Optical flow models and OpenCV's CLAHE expect data on a 0-255 scale, but
    rounding to uint8 at this point would discard the extra dynamic range that
    makes an uncompressed recording worth keeping in the first place.  Returning
    float32 preserves it.

    Args:
        frame: array of raw counts.
        bit_depth: sensor bit depth.  If None, inferred from the dtype and the
            observed maximum -- correct for the common 8/10/12/14/16-bit cases,
            but pass it explicitly if the recording is unusual.
    """
    if frame.dtype == np.uint8:
        return frame.astype(np.float32)

    if bit_depth is None:
        observed_max = int(frame.max()) if frame.size else 0
        for candidate in (8, 10, 12, 14, 16):
            if observed_max < (1 << candidate):
                bit_depth = candidate
                break
        else:
            bit_depth = 16

    full_scale = float((1 << int(bit_depth)) - 1)
    scaled = frame.astype(np.float32) * (255.0 / full_scale)
    return np.clip(scaled, 0.0, 255.0)


def apply_clahe(frame, clip_limit=2.0, tile_size=8, is_grayscale=False):
    """
    Contrast Limited Adaptive Histogram Equalization.

    OpenCV's CLAHE needs an integer image.  For float input the frame is
    promoted to uint16 first so the enhancement runs at better than 8-bit
    precision, then returned to the float 0-255 scale.

    Note that CLAHE is computed independently per frame.  That is a non-linear,
    spatially varying intensity map, so it weakens the brightness-constancy
    assumption both the tracker and the uncertainty estimator rely on -- most
    noticeably in flows whose global brightness genuinely fluctuates.  Compare
    results with and without it before adopting it.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))

    if is_grayscale:
        if np.issubdtype(frame.dtype, np.floating):
            as_u16 = np.clip(frame * 257.0, 0, 65535).astype(np.uint16)
            return clahe.apply(as_u16).astype(np.float32) / 257.0
        return clahe.apply(frame)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    lab_clahe = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


def prepare_bgr_to_tensor(frame, is_grayscale=False):
    """
    Convert a frame on the 0-255 scale to a (3, H, W) float tensor.

    Grayscale input is replicated across three channels, which is what the
    RAFT encoders expect.
    """
    if is_grayscale:
        frame_f = frame.astype(np.float32)
        img = np.stack([frame_f] * 3, axis=-1)
        return torch.from_numpy(img).permute(2, 0, 1)

    img = frame[..., ::-1].copy()
    return torch.from_numpy(img).permute(2, 0, 1).float()
