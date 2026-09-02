"""
Uncertainty propagation for time-averaged statistics.

The reference values here are not asserted from the formulas themselves -- they
are measured, by generating many noise realisations over a fixed true signal and
observing how much the estimators actually scatter. That is what the propagated
uncertainty is supposed to predict.
"""

import numpy as np
import pytest

from postprocess.velocity import _Accumulator, integral_timescale


def _measure_scatter(n_samples, sigma_noise, n_trials=600, seed=0):
    """Empirical scatter of the estimators, over noise only."""
    rng = np.random.default_rng(seed)
    truth = rng.normal(15.0, 1.0, n_samples)
    means, variances = [], []
    for _ in range(n_trials):
        u = truth + rng.normal(0.0, sigma_noise, n_samples)
        means.append(u.mean())
        variances.append(u.var())
    return truth, np.std(means), np.std(variances), np.mean(variances)


def _propagate(truth, sigma_noise, seed=1):
    """Run one realisation through the production accumulator."""
    rng = np.random.default_rng(seed)
    n = len(truth)
    err = rng.normal(0.0, sigma_noise, n)
    u = (truth + err).reshape(n, 1)
    v = np.zeros_like(u)
    var = np.full_like(u, sigma_noise ** 2)
    acc = _Accumulator((1,))
    acc.add(u, v, var, var, err.reshape(n, 1), np.zeros_like(u), np.ones_like(u))
    return acc.reduce()


def test_reynolds_stress_uncertainty_matches_measured_scatter():
    """sigma(<u'u'>) must predict how much <u'u'> actually moves under noise."""
    for n in (200, 2000):
        truth, _, scatter_uu, _ = _measure_scatter(n, 0.30)
        out = _propagate(truth, 0.30)
        assert out['uu_uncert'][0] == pytest.approx(scatter_uu, rel=0.25)


def test_reynolds_stress_uncertainty_shrinks_as_one_over_sqrt_n():
    small = _propagate(_measure_scatter(200, 0.30)[0], 0.30)['uu_uncert'][0]
    large = _propagate(_measure_scatter(2000, 0.30)[0], 0.30)['uu_uncert'][0]
    assert small / large == pytest.approx(np.sqrt(10.0), rel=0.35)


def test_noise_inflates_the_measured_variance_by_sigma_squared():
    """E[<u'u'>_measured] = <u'u'>_true + sigma^2, so the correction removes it."""
    truth, _, _, mean_measured = _measure_scatter(4000, 0.40)
    assert mean_measured == pytest.approx(truth.var() + 0.40 ** 2, abs=0.03)

    out = _propagate(truth, 0.40)
    raw_error = abs(out['uu'][0] - truth.var())
    corrected_error = abs(out['uu_noise_corrected'][0] - truth.var())
    assert corrected_error < raw_error


def test_bias_survives_averaging_and_random_error_does_not():
    n = 4000
    rng = np.random.default_rng(3)
    truth = rng.normal(15.0, 1.0, n).reshape(n, 1)
    ones = np.ones_like(truth)

    biased = _Accumulator((1,))
    biased.add(truth, np.zeros_like(truth), np.zeros_like(truth), np.zeros_like(truth),
               np.full_like(truth, 0.5), np.zeros_like(truth), ones)
    out_biased = biased.reduce()

    noisy = _Accumulator((1,))
    noise = rng.normal(0.0, 0.5, (n, 1))
    noisy.add(truth, np.zeros_like(truth), np.zeros_like(truth), np.zeros_like(truth),
              noise, np.zeros_like(truth), ones)
    out_noisy = noisy.reduce()

    assert out_biased['mean_u_bias'][0] == pytest.approx(0.5, abs=0.01)
    assert out_biased['mean_u_uncert'][0] == pytest.approx(0.5, abs=0.02)
    assert abs(out_noisy['mean_u_bias'][0]) < 0.05
    assert out_noisy['mean_u_uncert'][0] < 0.1


def test_mean_uncertainty_decreases_with_more_samples():
    """Random error averages down; the previous implementation kept it constant."""
    short = _propagate(_measure_scatter(200, 0.3)[0], 0.3)['mean_u_uncert'][0]
    long = _propagate(_measure_scatter(5000, 0.3)[0], 0.3)['mean_u_uncert'][0]
    assert long < short / 3.0


def test_invalid_samples_are_excluded():
    n = 100
    u = np.concatenate([np.full((n, 1), 10.0), np.full((n, 1), 1e6)])
    z = np.zeros_like(u)
    w = np.concatenate([np.ones((n, 1)), np.zeros((n, 1))])
    out = _Accumulator((1,))
    out.add(u, z, z, z, z, z, w)
    res = out.reduce()
    assert res['mean_u'][0] == pytest.approx(10.0)
    assert res['n_samples'][0] == n


def test_all_invalid_yields_nan_not_zero():
    z = np.zeros((10, 1))
    acc = _Accumulator((1,))
    acc.add(z, z, z, z, z, z, np.zeros_like(z))
    assert np.isnan(acc.reduce()['mean_u'][0])


def test_integral_timescale_detects_correlation():
    rng = np.random.default_rng(5)
    white = rng.normal(size=(4000, 1))
    tau_white = integral_timescale(white)[0]

    alpha = np.exp(-1.0 / 20.0)
    correlated = np.empty((4000, 1))
    correlated[0] = white[0]
    for i in range(1, 4000):
        correlated[i] = alpha * correlated[i - 1] + np.sqrt(1 - alpha ** 2) * white[i]
    tau_corr = integral_timescale(correlated)[0]

    assert tau_white < 3.0
    assert tau_corr > 5.0 * tau_white
