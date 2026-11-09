"""
Dense per-pixel uncertainty estimation for optical flow fields.

The estimator warps frame 2 by the predicted flow and searches a local window of
integer pixel shifts for the peak of the Normalized Cross-Correlation (NCC)
surface.  The offset between that peak and the zero-shift position is the
*residual displacement error* -- how far the tracker's answer sits from where
correlation analysis says the true match lies.

Two properties of that quantity matter for downstream use:

  * It is a **signed, per-frame realised error**, not a standard deviation.  The
    sign convention is ``flow_corrected = flow - residual``: a residual of -1 px
    means the tracker under-predicted the displacement by 1 px.  Part
    of it is systematic (a smoothness prior under-predicting the same gradient
    every frame) and part is random.  The sign is therefore kept: averaging the
    signed residual over time recovers the bias, and its temporal scatter
    recovers the random component.  See ``postprocess.velocity``.
  * It is only meaningful where the correlation surface is informative.  Points
    where the peak lies outside the search window, where the correlation is too
    weak to locate a peak, or where the frame border has been synthesised by
    padding are reported through the ``valid`` mask and must be excluded from
    statistics.
"""

from collections import namedtuple

import torch
import torch.nn.functional as F

#: Result of :func:`compute_uncertainty_batch`.
#:
#: sigma    -- (B, H, W, 2) float32, |residual| in pixels (unsigned magnitude)
#: residual -- (B, H, W, 2) float32, signed residual in pixels
#: valid    -- (B, H, W)    bool,    False where the estimate is uninformative
UncertaintyResult = namedtuple("UncertaintyResult", "sigma residual valid")

# Correlation peak below this value is treated as decorrelated (no information).
DEFAULT_MIN_PEAK_CORR = 0.3

# Floor applied to NCC values before the logarithm in the Gaussian peak fit.
_R_FLOOR = 1e-3


def smooth_field(f, window_size=32):
    """
    Box filter (moving average) over a square window, preserving spatial dims.

    Implemented as two 1D convolutions rather than one 2D convolution: a box
    filter is separable, so this costs O(k) instead of O(k^2) per pixel.
    """
    C = f.shape[1]
    pad_lo = window_size // 2
    pad_hi = window_size - pad_lo - 1

    kernel_v = torch.ones(C, 1, window_size, 1, device=f.device, dtype=f.dtype) / window_size
    kernel_h = torch.ones(C, 1, 1, window_size, device=f.device, dtype=f.dtype) / window_size

    f = F.conv2d(F.pad(f, (0, 0, pad_lo, pad_hi), mode='reflect'), kernel_v, groups=C)
    f = F.conv2d(F.pad(f, (pad_lo, pad_hi, 0, 0), mode='reflect'), kernel_h, groups=C)
    return f


def get_smoothed_ncc(img1, img2, window_size=32):
    """
    Normalized Cross-Correlation with local mean and variance over a window.

    Kept for backwards compatibility.  The batch estimator below uses
    :func:`_ncc_against_reference`, which hoists the frame-1 statistics out of
    the shift loop instead of recomputing them for every candidate shift.
    """
    mean1 = smooth_field(img1, window_size)
    var1 = smooth_field(img1 * img1, window_size) - mean1 * mean1
    return _ncc_against_reference(img1, mean1, var1, img2, window_size)


def _ncc_against_reference(img1, mean1, var1, img2, window_size):
    """NCC of *img2* against a reference whose local statistics are precomputed."""
    mean2 = smooth_field(img2, window_size)
    var2 = smooth_field(img2 * img2, window_size) - mean2 * mean2
    cov = smooth_field(img1 * img2, window_size) - mean1 * mean2

    std1 = torch.sqrt(torch.clamp(var1, min=1e-8))
    std2 = torch.sqrt(torch.clamp(var2, min=1e-8))
    return cov / (std1 * std2)


def warp_img(img, flow):
    """
    Warp an image tensor by a flow field.

    flow: (B, 2, H, W) in pixel displacement.
    """
    B, C, H, W = img.shape

    xx = torch.arange(0, W, device=img.device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=img.device).view(-1, 1).repeat(1, W)

    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1).float()
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1).float()

    grid = torch.cat((xx, yy), dim=1) + flow

    grid_x = 2.0 * grid[:, 0] / max(W - 1, 1) - 1.0
    grid_y = 2.0 * grid[:, 1] / max(H - 1, 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)  # (B, H, W, 2)

    return F.grid_sample(img, grid, align_corners=True, padding_mode='reflection')


def shift_img(img, dx, dy):
    """Shift an image tensor by integer pixel amounts using reflection padding."""
    B, C, H, W = img.shape
    pad_left = max(dx, 0)
    pad_right = max(-dx, 0)
    pad_top = max(dy, 0)
    pad_bottom = max(-dy, 0)

    padded = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')

    start_y = pad_top - dy
    start_x = pad_left - dx
    return padded[:, :, start_y:start_y + H, start_x:start_x + W]


def _to_grayscale(img):
    if img.shape[1] == 3:
        weight = torch.tensor([0.299, 0.587, 0.114], device=img.device).view(1, 3, 1, 1)
        return (img * weight).sum(dim=1, keepdim=True)
    return img


def compute_uncertainty_batch(image1, image2, flow, window_size=8, max_shift=None,
                              min_peak_corr=DEFAULT_MIN_PEAK_CORR):
    """
    Estimate the per-pixel residual displacement error via NCC peak shift.

    Args:
        image1, image2: (B, C, H, W) frame tensors.
        flow:           (B, 2, H, W) predicted displacement in pixels.
        window_size:    NCC correlation window, in pixels.
        max_shift:      Half-width of the integer shift search, in pixels.
                        Defaults to ``window_size // 2``.  Residual errors larger
                        than this cannot be measured and are flagged invalid, so
                        it should exceed the expected error -- which grows with
                        the per-frame displacement.
        min_peak_corr:  Correlation peaks below this are treated as decorrelated.

    Returns:
        :class:`UncertaintyResult` with ``sigma`` (B, H, W, 2) unsigned,
        ``residual`` (B, H, W, 2) signed, and ``valid`` (B, H, W) boolean.
        All arrays are numpy, displacements in pixels.

        The signed residual follows ``flow_corrected = flow - residual``.
    """
    image1 = _to_grayscale(image1)
    image2 = _to_grayscale(image2)

    img2_w = warp_img(image2, flow)

    if max_shift is None:
        max_shift = window_size // 2
    max_shift = max(int(max_shift), 1)

    shifts = range(-max_shift, max_shift + 1)
    n_shifts = len(shifts)
    B, _, H, W = image1.shape
    device = image1.device

    # Frame-1 local statistics are identical for every candidate shift.
    mean1 = smooth_field(image1, window_size)
    var1 = smooth_field(image1 * image1, window_size) - mean1 * mean1

    ncc_map = torch.zeros((B, n_shifts, n_shifts, H, W), device=device)
    for i, dx in enumerate(shifts):
        for j, dy in enumerate(shifts):
            shifted = shift_img(img2_w, dx, dy)
            ncc = _ncc_against_reference(image1, mean1, var1, shifted, window_size)
            ncc_map[:, i, j] = torch.clamp(ncc, -1.0, 1.0)[:, 0]

    ncc_flat = ncc_map.view(B, n_shifts * n_shifts, H, W)
    max_idx = torch.argmax(ncc_flat, dim=1, keepdim=True)
    peak_x = max_idx // n_shifts   # index along dx
    peak_y = max_idx % n_shifts    # index along dy

    peak_corr = torch.gather(ncc_flat, 1, max_idx).squeeze(1)   # (B, H, W)

    # The peak sitting on the edge of the search window means the true peak is
    # outside it: the residual is larger than we can measure, not equal to the
    # clamped value.
    at_boundary = (
        (peak_x == 0) | (peak_x == n_shifts - 1) |
        (peak_y == 0) | (peak_y == n_shifts - 1)
    ).squeeze(1)

    # Clamp only so the 3-point fit has neighbours; boundary hits are flagged above.
    center_x = torch.clamp(peak_x, 1, n_shifts - 2)
    center_y = torch.clamp(peak_y, 1, n_shifts - 2)

    def peak_shift_1d(cx, cy, is_x):
        """Sub-pixel peak location along one axis via a 3-point Gaussian fit."""
        if is_x:
            idx_m = (cx - 1) * n_shifts + cy
            idx_0 = cx * n_shifts + cy
            idx_p = (cx + 1) * n_shifts + cy
        else:
            idx_m = cx * n_shifts + (cy - 1)
            idx_0 = cx * n_shifts + cy
            idx_p = cx * n_shifts + (cy + 1)

        r_m = torch.gather(ncc_flat, 1, idx_m).squeeze(1)
        r_0 = torch.gather(ncc_flat, 1, idx_0).squeeze(1)
        r_p = torch.gather(ncc_flat, 1, idx_p).squeeze(1)

        # A Gaussian fit is only defined on a genuine local maximum.
        convex = (r_0 > r_m) & (r_0 > r_p)

        log_m = torch.log(torch.clamp(r_m, _R_FLOOR, 1.0))
        log_0 = torch.log(torch.clamp(r_0, _R_FLOOR, 1.0))
        log_p = torch.log(torch.clamp(r_p, _R_FLOOR, 1.0))

        den = 2.0 * (log_m + log_p - 2.0 * log_0)
        den = torch.where(den.abs() < 1e-8, torch.full_like(den, -1e-8), den)
        sub = torch.clamp((log_m - log_p) / den, -1.0, 1.0)

        discrete = (cx if is_x else cy).squeeze(1).float() - max_shift
        return discrete + sub, convex

    delta_u, convex_u = peak_shift_1d(center_x, center_y, is_x=True)
    delta_v, convex_v = peak_shift_1d(center_x, center_y, is_x=False)

    # Reflection padding in warp_img/shift_img synthesises correlation near the
    # frame border, which biases the estimate low there.
    margin = max_shift + window_size
    near_edge = torch.zeros((B, H, W), dtype=torch.bool, device=device)
    if 2 * margin < min(H, W):
        near_edge[:, :margin, :] = True
        near_edge[:, -margin:, :] = True
        near_edge[:, :, :margin] = True
        near_edge[:, :, -margin:] = True
    else:
        near_edge[:] = True

    valid = (~at_boundary) & (peak_corr >= min_peak_corr) & convex_u & convex_v & (~near_edge)

    residual = torch.stack((delta_u, delta_v), dim=-1)
    sigma = residual.abs()

    return UncertaintyResult(
        sigma=sigma.cpu().numpy(),
        residual=residual.cpu().numpy(),
        valid=valid.cpu().numpy(),
    )
