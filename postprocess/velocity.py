"""
Time-averaged velocity statistics and their uncertainties.

Uncertainty model
-----------------
The tracker stores a signed per-frame residual displacement error, ``d(t)``.
Averaged over time it decomposes into a systematic and a random part::

    d(t) = b + e(t),   b = mean_t d(t),   Var(e) = sigma_rand^2

The two propagate differently and are reported separately:

* ``b`` is a bias.  It does not shrink with record length, and it displaces the
  mean rather than widening it.
* ``e(t)`` is random.  Its effect on a time average shrinks as
  ``1 / sqrt(N_eff)``, where ``N_eff = N / (2 * tau_int)`` accounts for the
  measurements being autocorrelated in time.

For statistics of the fluctuations, random measurement error also *inflates* the
apparent variance: ``E[<u'u'>_measured] = <u'u'>_true + sigma_rand^2``.  That
noise floor is subtracted, and both corrected and uncorrected values are stored
so the size of the correction stays visible.

Files written before the signed residual existed are still readable: the
unsigned ``uncertainty`` dataset is then treated as wholly random, which is the
conservative reading.
"""

import os

import h5py
import numpy as np
from tqdm import tqdm

from pipeline_utils.io import (block_centers, legacy_block_centers,
                               uses_legacy_block_centers)

CHUNK = 100

#: Frames used to estimate the integral timescale.  It is a statistical property
#: that converges long before a full high-speed record is exhausted, and
#: buffering the whole record would cost gigabytes.  Frames must be contiguous:
#: subsampling in time would change what a lag means.
TIMESCALE_MAX_FRAMES = 4000

#: Spatial stride for the field-wide timescale estimate.  The timescale varies
#: smoothly across the grid, so it is computed on a coarse subsample and
#: expanded back, rather than held at full resolution for every point.
TIMESCALE_STRIDE = 4

#: Attributes copied from a source file into every derived file.
PROVENANCE_KEYS = (
    'mm_per_px', 'fps_capture', 'window_width', 'window_height', 'roi',
    'throat_loc_px', 'rotate_angle', 'method', 'model', 'raft_iters',
    'uncert_window', 'uncert_max_shift', 'use_clahe', 'clahe_clip_limit',
    'clahe_tile_size', 'bit_depth', 'n_frames', 'source_video', 'code_version',
)


def compute_spatial_derivatives(u, v, dx, dy):
    """
    Spatial derivatives of a 2D velocity field.

    Divergence is a useful ground-truth-free error indicator: in a single-phase
    incompressible region mass conservation requires it to vanish, so the
    measured value bounds the error.

    Returns a dict with divergence, vorticity, the strain-rate tensor's shear
    component, and the four velocity gradients.
    """
    du_dy, du_dx = np.gradient(u, dy, dx)
    dv_dy, dv_dx = np.gradient(v, dy, dx)

    return {
        'divergence': du_dx + dv_dy,
        'vorticity': dv_dx - du_dy,
        'shear_strain_rate': 0.5 * (du_dy + dv_dx),
        'du_dx': du_dx, 'du_dy': du_dy, 'dv_dx': dv_dx, 'dv_dy': dv_dy,
    }


def extract_profile_slice(x_query, x_grid, field_data):
    """Extract the column of *field_data* nearest to *x_query*."""
    return field_data[:, int(np.abs(x_grid - x_query).argmin())]


def rotate_velocity(u, v, angle_deg):
    """Rotate a velocity vector field by *angle_deg*."""
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return u * c - v * s, u * s + v * c


def copy_provenance(src_attrs, dest):
    """Copy calibration and provenance attributes into a derived file."""
    for key in PROVENANCE_KEYS:
        if key in src_attrs:
            dest.attrs[key] = src_attrs[key]


def integral_timescale(series, axis=0, max_lag=200):
    """
    Integral timescale of a signal, in samples, from its autocorrelation.

    Integrated up to the first zero crossing (Benedict & Gould, Exp. Fluids 22,
    1996).  Used to convert a sample count into an effective number of
    independent samples.

    Args:
        series: (T, ...) array, mean removed internally.
        max_lag: highest lag evaluated; the timescale is capped there.
    """
    x = np.moveaxis(np.asarray(series, dtype=np.float64), axis, 0)
    n = x.shape[0]
    if n < 4:
        return np.ones(x.shape[1:], dtype=np.float64)

    x = x - x.mean(axis=0, keepdims=True)
    denom = np.sum(x * x, axis=0)
    denom = np.where(denom <= 0, np.inf, denom)

    tau = np.full(x.shape[1:], 0.5, dtype=np.float64)
    still_positive = np.ones(x.shape[1:], dtype=bool)

    for lag in range(1, min(max_lag, n - 1) + 1):
        rho = np.sum(x[lag:] * x[:-lag], axis=0) / denom
        still_positive &= rho > 0
        if not still_positive.any():
            break
        tau += np.where(still_positive, rho, 0.0)

    return np.maximum(tau, 0.5)


def _read_uncertainty(f, t0, t1, shape):
    """
    Read residual, noise variance and validity for a chunk.

    Returns (residual, var, valid) with residual signed in pixels, var the
    squared unsigned uncertainty in pixels, and valid a boolean mask.  Any of
    the three may be synthesised when the source file predates them.
    """
    if 'residual' in f:
        residual = f['residual'][t0:t1].astype(np.float64)
    elif 'uncertainty' in f:
        # Unsigned only: treat it as wholly random (conservative) with no bias.
        residual = np.zeros(shape + (2,), dtype=np.float64)
    else:
        residual = np.zeros(shape + (2,), dtype=np.float64)

    if 'uncertainty' in f:
        var = f['uncertainty'][t0:t1].astype(np.float64) ** 2
    else:
        var = np.zeros(shape + (2,), dtype=np.float64)

    if 'valid_fraction' in f:
        valid = f['valid_fraction'][t0:t1].astype(np.float64) > 0.5
    else:
        valid = np.ones(shape, dtype=bool)

    return residual, var, valid


class _Accumulator:
    """Running masked sums for the mean, the stresses and the uncertainty terms."""

    FIELDS = ('n', 'u', 'v', 'uu', 'vv', 'uv',
              'var_u', 'var_v', 'res_u', 'res_v', 'res_u2', 'res_v2',
              'u_var_u', 'u2_var_u', 'v_var_v', 'v2_var_v',
              'u_var_v', 'u2_var_v', 'v_var_u', 'v2_var_u')

    def __init__(self, shape):
        for name in self.FIELDS:
            setattr(self, name, np.zeros(shape, dtype=np.float64))
        #: Whether any signed residual was supplied.  Without it the bias and
        #: the random part of the error cannot be told apart, and several
        #: quantities below are not computable.
        self.signed = False

    def add(self, u, v, var_u, var_v, res_u, res_v, w):
        """Add a chunk. *w* is a 0/1 weight of the same shape as *u*."""
        if np.any(res_u) or np.any(res_v):
            self.signed = True
        s = lambda a: np.sum(a * w, axis=0, dtype=np.float64)
        self.n += s(np.ones_like(u))
        self.u += s(u)
        self.v += s(v)
        self.uu += s(u * u)
        self.vv += s(v * v)
        self.uv += s(u * v)
        self.var_u += s(var_u)
        self.var_v += s(var_v)
        self.res_u += s(res_u)
        self.res_v += s(res_v)
        self.res_u2 += s(res_u * res_u)
        self.res_v2 += s(res_v * res_v)
        self.u_var_u += s(u * var_u)
        self.u2_var_u += s(u * u * var_u)
        self.v_var_v += s(v * var_v)
        self.v2_var_v += s(v * v * var_v)
        self.u_var_v += s(u * var_v)
        self.u2_var_v += s(u * u * var_v)
        self.v_var_u += s(v * var_u)
        self.v2_var_u += s(v * v * var_u)

    def reduce(self, tau_u=None, tau_v=None):
        """Turn the running sums into statistics and their uncertainties."""
        n = np.maximum(self.n, 1.0)
        empty = self.n < 1

        with np.errstate(invalid='ignore', divide='ignore'):
            umean = self.u / n
            vmean = self.v / n
            uu = self.uu / n - umean ** 2
            vv = self.vv / n - vmean ** 2
            uv = self.uv / n - umean * vmean

            # Bias / random split of the signed residual.
            bias_u = self.res_u / n
            bias_v = self.res_v / n
            var_rand_u = np.maximum(self.res_u2 / n - bias_u ** 2, 0.0)
            var_rand_v = np.maximum(self.res_v2 / n - bias_v ** 2, 0.0)

            # Without the sign, bias and scatter cannot be separated.  The
            # unsigned magnitude is all there is, so it stands in for the random
            # part -- adequate for the uncertainty of the mean, but *not* for the
            # noise-floor subtraction below, which it would over-subtract by
            # however much of the residual is actually systematic.
            # The bias *term* and the reported bias *field* differ when the sign
            # is missing: the term drops out of the sum (an unknown contribution
            # cannot be added), while the field is reported as NaN so nobody
            # mistakes "not measured" for "measured to be zero".
            bias_u_term, bias_v_term = bias_u, bias_v
            if not self.signed:
                var_rand_u = self.var_u / n
                var_rand_v = self.var_v / n
                bias_u_term = np.zeros_like(bias_u)
                bias_v_term = np.zeros_like(bias_v)
                bias_u = np.full_like(bias_u, np.nan)
                bias_v = np.full_like(bias_v, np.nan)

            tau_u = np.ones_like(n) if tau_u is None else np.maximum(tau_u, 0.5)
            tau_v = np.ones_like(n) if tau_v is None else np.maximum(tau_v, 0.5)
            n_eff_u = np.maximum(n / (2.0 * tau_u), 1.0)
            n_eff_v = np.maximum(n / (2.0 * tau_v), 1.0)

            # Uncertainty of the mean: a bias that survives averaging, plus a
            # random part that shrinks with the effective sample count.
            umean_uncert = np.sqrt(bias_u_term ** 2 + var_rand_u / n_eff_u)
            vmean_uncert = np.sqrt(bias_v_term ** 2 + var_rand_v / n_eff_v)

            # Instantaneous measurement precision: a different quantity, kept
            # because it answers "how well is one frame resolved?".
            precision_u = np.sqrt(self.var_u / n)
            precision_v = np.sqrt(self.var_v / n)

            # Random noise inflates the apparent variance; remove that floor.
            # Only meaningful with signed residuals: an unsigned magnitude is
            # dominated by bias, which contributes nothing to variance, so
            # subtracting it drives the result negative.
            if self.signed:
                uu_corr = uu - var_rand_u
                vv_corr = vv - var_rand_v
            else:
                uu_corr = np.full_like(uu, np.nan)
                vv_corr = np.full_like(vv, np.nan)

            # Delta method on the stresses: Var(R) = (4/N^2) * SUM (u-ubar)^2 var_u
            var_uu = (4.0 / n ** 2) * (self.u2_var_u - 2 * umean * self.u_var_u
                                       + umean ** 2 * self.var_u)
            var_vv = (4.0 / n ** 2) * (self.v2_var_v - 2 * vmean * self.v_var_v
                                       + vmean ** 2 * self.var_v)
            part1 = self.v2_var_u - 2 * vmean * self.v_var_u + vmean ** 2 * self.var_u
            part2 = self.u2_var_v - 2 * umean * self.u_var_v + umean ** 2 * self.var_v
            var_uv = (1.0 / n ** 2) * (part1 + part2)

        out = {
            'mean_u': umean, 'mean_v': vmean,
            'uu': uu, 'vv': vv, 'uv': uv,
            'uu_noise_corrected': uu_corr, 'vv_noise_corrected': vv_corr,
            'mean_u_uncert': umean_uncert, 'mean_v_uncert': vmean_uncert,
            'mean_u_bias': bias_u, 'mean_v_bias': bias_v,
            'precision_u': precision_u, 'precision_v': precision_v,
            'uu_uncert': np.sqrt(np.maximum(var_uu, 0)),
            'vv_uncert': np.sqrt(np.maximum(var_vv, 0)),
            'uv_uncert': np.sqrt(np.maximum(var_uv, 0)),
            'n_samples': self.n, 'n_effective': n_eff_u,
            'has_signed_residual': self.signed,
        }
        for key, value in out.items():
            if key not in ('n_samples', 'has_signed_residual'):
                out[key] = np.where(empty, np.nan, value)
        return out


def _scaling(attrs):
    mm_per_px = float(attrs.get('mm_per_px', 1.0))
    fps = float(attrs.get('fps_capture', 1.0))
    return mm_per_px * 1e-3 * fps


def compute_time_average_fields(h5_filepath, frame_start=None, frame_end=None,
                                estimate_timescale=True):
    """
    Full-field time averages, Reynolds stresses and their uncertainties.

    Points flagged invalid by the tracker are excluded, so ``n_samples`` varies
    across the grid.
    """
    with h5py.File(h5_filepath, 'r') as f:
        vel = f['velocity']
        attrs = dict(vel.attrs)
        scale = _scaling(attrs)

        n_frames = vel.shape[0]
        start = max(0, frame_start) if frame_start is not None else 0
        end = min(n_frames, frame_end) if frame_end is not None else n_frames
        ny, nx = vel.shape[1:3]

        acc = _Accumulator((ny, nx))
        centre_series = []
        buffered = 0

        for t0 in tqdm(range(start, end, CHUNK), desc='Averaging fields'):
            t1 = min(t0 + CHUNK, end)
            chunk = vel[t0:t1].astype(np.float64)
            u = chunk[..., 0] * scale
            v = -chunk[..., 1] * scale     # image y grows downward; flip to physical

            residual, var, valid = _read_uncertainty(f, t0, t1, chunk.shape[:3])
            w = valid.astype(np.float64)

            acc.add(u, v, var[..., 0] * scale ** 2, var[..., 1] * scale ** 2,
                    residual[..., 0] * scale, -residual[..., 1] * scale, w)

            if estimate_timescale and buffered < TIMESCALE_MAX_FRAMES:
                centre_series.append(u[:, ::TIMESCALE_STRIDE, ::TIMESCALE_STRIDE])
                buffered += u.shape[0]

        tau = None
        if estimate_timescale and centre_series:
            coarse = integral_timescale(np.concatenate(centre_series, axis=0))
            tau = np.repeat(np.repeat(coarse, TIMESCALE_STRIDE, axis=0),
                            TIMESCALE_STRIDE, axis=1)[:ny, :nx]

        results = acc.reduce(tau_u=tau, tau_v=tau)

    return results


def extract_line_profiles(h5_filepath, x_positions_mm, angle_deg=0.0, frame_idx=None,
                          output_path=None, estimate_timescale=True):
    """
    Extract wall-normal profiles at the requested downstream stations.

    Stations are given in mm relative to ``throat_loc_px``.  Profiles are taken
    as vertical columns, linearly interpolated between the two nearest grid
    columns; the field is assumed already aligned by ``--rotate_angle`` at
    tracking time.

    Writes a ``*_lines.h5`` carrying the profiles, their uncertainties, and the
    calibration and provenance of the source file.
    """
    if abs(angle_deg) > 1e-9:
        raise NotImplementedError(
            "angle_deg is not implemented. This extractor assumes the velocity "
            "field was aligned during tracking; re-run tracker.py with "
            "--rotate_angle instead.")

    with h5py.File(h5_filepath, 'r') as f:
        vel = f['velocity']
        attrs = dict(vel.attrs)
        scale = _scaling(attrs)

        mm_per_px = float(attrs.get('mm_per_px', 1.0))
        win_w = int(attrs.get('window_width', 1))
        win_h = int(attrs.get('window_height', 1))
        n_frames, ny, nx = vel.shape[:3]

        roi = list(attrs.get('roi', [0, -1, 0, -1]))
        if roi[1] == -1:
            roi[1] = roi[0] + ny * win_h
        if roi[3] == -1:
            roi[3] = roi[2] + nx * win_w

        throat = list(attrs['throat_loc_px']) if 'throat_loc_px' in attrs else [roi[1], 0]

        # Block centres, matching however this file was actually written.
        centers = legacy_block_centers if uses_legacy_block_centers(attrs) else block_centers
        x_px = roi[2] + centers(nx, win_w)
        y_px = roi[0] + centers(ny, win_h)
        xmm = (x_px - throat[1]) * mm_per_px
        ymm = (throat[0] - y_px) * mm_per_px

        lines = []
        for x_wall in x_positions_mm:
            j_frac = np.interp(x_wall, xmm, np.arange(nx))
            j0 = int(np.floor(j_frac))
            j1 = min(j0 + 1, nx - 1)
            frac = j_frac - j0
            lines.append({'x_wall': x_wall, 'j0': j0, 'j1': j1,
                          'w0': 1.0 - frac, 'w1': frac})

        if frame_idx is not None:
            start, end = max(0, frame_idx), min(n_frames, frame_idx + 1)
        else:
            start, end = 0, n_frames

        accs = [_Accumulator((ny,)) for _ in lines]
        series = [[] for _ in lines]
        buffered = 0

        for t0 in tqdm(range(start, end, CHUNK), desc='Extracting profiles'):
            t1 = min(t0 + CHUNK, end)
            chunk = vel[t0:t1].astype(np.float64)
            residual, var, valid = _read_uncertainty(f, t0, t1, chunk.shape[:3])

            for li, info in enumerate(lines):
                j0, j1, w0, w1 = info['j0'], info['j1'], info['w0'], info['w1']

                def col(a, c):
                    return a[:, :, j0, c] * w0 + a[:, :, j1, c] * w1

                u = col(chunk, 0) * scale
                v = -col(chunk, 1) * scale
                # Interpolate variance with squared weights: the two columns are
                # combined linearly, so their independent errors add in quadrature.
                var_u = (var[:, :, j0, 0] * w0 ** 2 + var[:, :, j1, 0] * w1 ** 2) * scale ** 2
                var_v = (var[:, :, j0, 1] * w0 ** 2 + var[:, :, j1, 1] * w1 ** 2) * scale ** 2
                res_u = col(residual, 0) * scale
                res_v = -col(residual, 1) * scale
                w = (valid[:, :, j0] & valid[:, :, j1]).astype(np.float64)

                accs[li].add(u, v, var_u, var_v, res_u, res_v, w)
                if estimate_timescale and frame_idx is None and buffered < TIMESCALE_MAX_FRAMES:
                    series[li].append(u)
            if estimate_timescale and frame_idx is None:
                buffered += chunk.shape[0]

        per_line = []
        for li in range(len(lines)):
            tau = None
            if estimate_timescale and series[li]:
                tau = integral_timescale(np.concatenate(series[li], axis=0))
            per_line.append(accs[li].reduce(tau_u=tau, tau_v=tau))

        if output_path is None:
            stem = os.path.splitext(h5_filepath)[0]
            suffix = f'_frame_{frame_idx}_lines.h5' if frame_idx is not None else '_lines.h5'
            output_path = stem + suffix

        instantaneous = frame_idx is not None

        with h5py.File(output_path, 'w') as fout:
            copy_provenance(attrs, fout)
            fout.attrs['coordinate_type'] = 'wall_normal'
            fout.attrs['source_file'] = os.path.basename(h5_filepath)
            fout.attrs['instantaneous'] = int(instantaneous)
            signed = bool(per_line[0]['has_signed_residual']) if per_line else False
            fout.attrs['has_signed_residual'] = int(signed)
            if not signed:
                fout.attrs['uncertainty_note'] = (
                    'Source file carries only the unsigned residual magnitude, so '
                    'bias and random error could not be separated. The noise-floor '
                    'correction and the bias field are therefore absent, and '
                    'mean_velocity_uncertainty treats the whole residual as random '
                    '- which understates it to the extent the residual is '
                    'systematic. Re-track to obtain the signed residual.')
            if instantaneous:
                fout.attrs['frame_index'] = frame_idx

            for li, info in enumerate(lines):
                res = per_line[li]
                grp = fout.create_group(
                    f"x_location_{str(info['x_wall']).replace('.', '_')}")
                grp.attrs['x_mm'] = float(info['x_wall'])
                grp.attrs['coordinate_type'] = 'wall_normal_distance'

                grp.create_dataset('y_coordinates', data=ymm)
                grp.create_dataset('n_samples', data=res['n_samples'])

                vel_ds = grp.create_dataset(
                    'mean_velocity',
                    data=np.stack([res['mean_u'], res['mean_v']], axis=1))
                vel_ds.attrs['components'] = ['u_parallel', 'v_normal']
                vel_ds.attrs['units'] = 'm/s'

                unc_ds = grp.create_dataset(
                    'mean_velocity_uncertainty',
                    data=np.stack([res['mean_u_uncert'], res['mean_v_uncert']], axis=1))
                unc_ds.attrs['components'] = ['sigma_u_parallel', 'sigma_v_normal']
                unc_ds.attrs['definition'] = 'sqrt(bias^2 + var_random / N_eff)'

                if res['has_signed_residual']:
                    grp.create_dataset(
                        'mean_velocity_bias',
                        data=np.stack([res['mean_u_bias'], res['mean_v_bias']], axis=1))
                grp.create_dataset(
                    'instantaneous_precision',
                    data=np.stack([res['precision_u'], res['precision_v']], axis=1))

                if not instantaneous:
                    rey = grp.create_dataset(
                        'reynolds_stresses',
                        data=np.stack([res['uu'], res['uv'], res['vv']], axis=1))
                    rey.attrs['components'] = ['uu', 'uv', 'vv']
                    rey.attrs['units'] = 'm^2/s^2'
                    rey.attrs['noise_corrected'] = 0

                    if res['has_signed_residual']:
                        corr = grp.create_dataset(
                            'reynolds_stresses_noise_corrected',
                            data=np.stack([res['uu_noise_corrected'], res['uv'],
                                           res['vv_noise_corrected']], axis=1))
                        corr.attrs['components'] = ['uu', 'uv', 'vv']
                        corr.attrs['noise_corrected'] = 1

                    rey_unc = grp.create_dataset(
                        'reynolds_stresses_uncertainty',
                        data=np.stack([res['uu_uncert'], res['uv_uncert'],
                                       res['vv_uncert']], axis=1))
                    rey_unc.attrs['components'] = ['sigma_uu', 'sigma_uv', 'sigma_vv']
                    grp.create_dataset('n_effective', data=res['n_effective'])

        results = {}
        for li, info in enumerate(lines):
            res = per_line[li]
            results[info['x_wall']] = dict(res, y_coords=ymm)

        return results, output_path
