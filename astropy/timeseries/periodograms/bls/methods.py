# Licensed under a 3-clause BSD style license - see LICENSE.rst

__all__ = ["bls_fast", "bls_slow"]

from functools import partial

import numpy as np

from ._impl import bls_impl
from .window import transit_fraction


def bls_slow(t, y, ivar, period, duration, oversample, use_likelihood):
    """Compute the periodogram using a brute force reference method.

    t : array-like
        Sequence of observation times.
    y : array-like
        Sequence of observations associated with times t.
    ivar : array-like
        The inverse variance of ``y``.
    period : array-like
        The trial periods where the periodogram should be computed.
    duration : array-like
        The durations that should be tested.
    oversample :
        The resolution of the phase grid in units of durations.
    use_likeliood : bool
        If true, maximize the log likelihood over phase, duration, and depth.

    Returns
    -------
    power : array-like
        The periodogram evaluated at the periods in ``period``.
    depth : array-like
        The estimated depth of the maximum power model at each period.
    depth_err : array-like
        The 1-sigma uncertainty on ``depth``.
    duration : array-like
        The maximum power duration at each period.
    transit_time : array-like
        The maximum power phase of the transit in units of time. This
        indicates the mid-transit time and it will always be in the range
        (0, period).
    depth_snr : array-like
        The signal-to-noise with which the depth is measured at maximum power.
    log_likelihood : array-like
        The log likelihood of the maximum power model.

    """
    f = partial(_bls_slow_one, t, y, ivar, duration, oversample, use_likelihood)
    return _apply(f, period)


def bls_fast(t, y, ivar, period, duration, oversample, use_likelihood):
    """Compute the periodogram using an optimized Cython implementation.

    t : array-like
        Sequence of observation times.
    y : array-like
        Sequence of observations associated with times t.
    ivar : array-like
        The inverse variance of ``y``.
    period : array-like
        The trial periods where the periodogram should be computed.
    duration : array-like
        The durations that should be tested.
    oversample :
        The resolution of the phase grid in units of durations.
    use_likeliood : bool
        If true, maximize the log likelihood over phase, duration, and depth.

    Returns
    -------
    power : array-like
        The periodogram evaluated at the periods in ``period``.
    depth : array-like
        The estimated depth of the maximum power model at each period.
    depth_err : array-like
        The 1-sigma uncertainty on ``depth``.
    duration : array-like
        The maximum power duration at each period.
    transit_time : array-like
        The maximum power phase of the transit in units of time. This
        indicates the mid-transit time and it will always be in the range
        (0, period).
    depth_snr : array-like
        The signal-to-noise with which the depth is measured at maximum power.
    log_likelihood : array-like
        The log likelihood of the maximum power model.

    """
    return bls_impl(t, y, ivar, period, duration, oversample, use_likelihood)


def bls_integrated(
    t, y, ivar, exposure_time, period, duration, oversample, use_likelihood
):
    """Compute the periodogram using exact exposure-window averages."""
    f = partial(
        _bls_integrated_one,
        t,
        y,
        ivar,
        exposure_time,
        duration,
        oversample,
        use_likelihood,
    )
    return _apply(f, period)


def weighted_box_stats(y, ivar, fraction):
    """Fit a constant plus a box shape with exposure-overlap fractions."""
    weight = ivar
    weighted_y = weight * y
    w = np.sum(weight)
    wq = np.sum(weight * fraction)
    wqq = np.sum(weight * fraction * fraction)
    wy = np.sum(weighted_y)
    wqy = np.sum(weighted_y * fraction)
    determinant = w * wqq - wq * wq

    if determinant <= 0.0:
        baseline = wy / w if w > 0.0 else 0.0
        return 0.0, baseline, np.inf, 0.0

    baseline = (wy * wqq - wqy * wq) / determinant
    depth = (wq * wy - w * wqy) / determinant
    depth_var = w / determinant
    log_likelihood = 0.5 * depth * depth / depth_var if depth_var > 0.0 else 0.0
    return depth, baseline, np.sqrt(depth_var), log_likelihood


def _bls_slow_one(t, y, ivar, duration, oversample, use_likelihood, period):
    """A private function to compute the brute force periodogram result."""
    best = (-np.inf, None)
    hp = 0.5 * period
    min_t = np.min(t)
    for dur in duration:
        # Compute the phase grid (this is set by the duration and oversample).
        d_phase = dur / oversample
        phase = np.arange(0, period + d_phase, d_phase)

        for t0 in phase:
            # Figure out which data points are in and out of transit.
            m_in = np.abs((t - min_t - t0 + hp) % period - hp) < 0.5 * dur
            m_out = ~m_in

            # Compute the estimates of the in and out-of-transit flux.
            ivar_in = np.sum(ivar[m_in])
            ivar_out = np.sum(ivar[m_out])
            y_in = np.sum(y[m_in] * ivar[m_in]) / ivar_in
            y_out = np.sum(y[m_out] * ivar[m_out]) / ivar_out

            # Use this to compute the best fit depth and uncertainty.
            depth = y_out - y_in
            depth_err = np.sqrt(1.0 / ivar_in + 1.0 / ivar_out)
            snr = depth / depth_err

            # Compute the log likelihood of this model.
            loglike = -0.5 * np.sum((y_in - y[m_in]) ** 2 * ivar[m_in])
            loglike += 0.5 * np.sum((y_out - y[m_in]) ** 2 * ivar[m_in])

            # Choose which objective should be used for the optimization.
            if use_likelihood:
                objective = loglike
            else:
                objective = snr

            # If this model is better than any before, keep it.
            if depth > 0 and objective > best[0]:
                best = (
                    objective,
                    (
                        objective,
                        depth,
                        depth_err,
                        dur,
                        (t0 + min_t) % period,
                        snr,
                        loglike,
                    ),
                )

    return best[1]


def _bls_integrated_one(
    t, y, ivar, exposure_time, duration, oversample, use_likelihood, period
):
    """Compute the exact integrated result for one period."""
    best = (-np.inf, None)
    min_t = np.min(t)
    for trial_duration in duration:
        phase_step = trial_duration / oversample
        phase = np.arange(0, period, phase_step)

        for transit_time in phase:
            fraction = transit_fraction(
                t,
                exposure_time,
                period,
                trial_duration,
                min_t + transit_time,
            )
            depth, _, depth_err, log_like = weighted_box_stats(y, ivar, fraction)
            snr = depth / depth_err if np.isfinite(depth_err) and depth_err > 0 else 0
            objective = log_like if use_likelihood else snr

            if depth > 0 and objective > best[0]:
                best = (
                    objective,
                    (
                        objective,
                        depth,
                        depth_err,
                        trial_duration,
                        transit_time,
                        snr,
                        log_like,
                    ),
                )

    return best[1]


def _apply(f, period):
    return tuple(map(np.array, zip(*map(f, period))))
