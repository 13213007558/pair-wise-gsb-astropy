# Licensed under a 3-clause BSD style license - see LICENSE.rst

__all__ = ["bls_fast", "bls_slow"]

from functools import partial

import numpy as np

from ._impl import bls_impl


def window_overlap_fraction(t, exptime, period, duration, transit_time):
    """Compute the fraction of each exposure window that falls in transit.

    The box-shaped transit light curve (unit depth, centered on
    `transit_time` with total duration `duration` and periodic with
    `period`) is averaged over the exposure window
    `[t - 0.5 * exptime, t + 0.5 * exptime]` of each observation. The
    result is the fraction of each window during which the transit is
    active. Observations with zero (or negative) exposure time are treated
    as instantaneous samples of the box light curve.

    The computation is exact for windows that straddle the phase zero
    point and for exposure times longer than one period.

    t : array-like
        Mid-exposure times of the observations.
    exptime : array-like
        Exposure duration for each observation (broadcast to `t`).
    period : float
        The period of the transit signal.
    duration : float
        The duration of the transit (must be shorter than `period`).
    transit_time : float
        The mid-transit time of a reference transit.

    Returns
    -------
    fraction : `~numpy.ndarray`
        The in-transit fraction of each exposure window, in the range
        [0, 1].

    """
    t = np.ascontiguousarray(t, dtype=np.float64)
    exptime = np.ascontiguousarray(exptime, dtype=np.float64)
    transit_time = np.ascontiguousarray(transit_time, dtype=np.float64)
    t, exptime, transit_time = np.broadcast_arrays(t, exptime, transit_time)

    half_period = 0.5 * period
    half_duration = 0.5 * duration

    def cumulative_box(x, tt):
        # Antiderivative of the periodic box light curve: the integrated
        # in-transit time accumulated up to `x`.
        shifted = x - tt + half_period
        n_periods = np.floor(shifted / period)
        phase = shifted - n_periods * period - half_period
        return n_periods * duration + np.clip(phase + half_duration, 0.0, duration)

    fraction = np.empty_like(t)
    instantaneous = exptime <= 0.0
    if np.any(instantaneous):
        tt = transit_time[instantaneous]
        phase = (t[instantaneous] - tt + half_period) % period - half_period
        fraction[instantaneous] = (np.abs(phase) < half_duration).astype(np.float64)
    if np.any(~instantaneous):
        e = exptime[~instantaneous]
        tt = transit_time[~instantaneous]
        lo = t[~instantaneous] - 0.5 * e
        hi = t[~instantaneous] + 0.5 * e
        fraction[~instantaneous] = (cumulative_box(hi, tt) - cumulative_box(lo, tt)) / e
    return fraction


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


def bls_exptime(t, y, ivar, exptime, period, duration, oversample, use_likelihood):
    """Compute the periodogram for observations with finite exposure times.

    Each observation is modeled as the average of the box light curve over
    its exposure window (see :func:`window_overlap_fraction`), and the
    depth is estimated by weighted least squares at every trial phase,
    period, and duration. For zero exposure times this reduces exactly to
    the statistics computed by :func:`bls_slow`.

    t : array-like
        Sequence of mid-exposure observation times.
    y : array-like
        Sequence of observations associated with times t.
    ivar : array-like
        The inverse variance of `y`.
    exptime : array-like
        The exposure duration for each observation.
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
        The periodogram evaluated at the periods in `period`.
    depth : array-like
        The estimated depth of the maximum power model at each period.
    depth_err : array-like
        The 1-sigma uncertainty on `depth`.
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
    f = partial(
        _bls_exptime_one, t, y, ivar, exptime, duration, oversample, use_likelihood
    )
    return _apply(f, period)


def _bls_exptime_one(t, y, ivar, exptime, duration, oversample, use_likelihood, period):
    """A private function to compute the exposure-aware periodogram."""
    best = (-np.inf, None)
    min_t = np.min(t)

    # Sums that do not depend on the trial transit parameters.
    s_w = np.sum(ivar)
    s_y = np.sum(ivar * y)
    ivar_y = ivar * y

    for dur in duration:
        # Compute the phase grid (this is set by the duration and oversample).
        d_phase = dur / oversample
        phase = np.arange(0, period + d_phase, d_phase)

        # The in-transit fraction of each exposure window for every trial
        # phase, with shape (len(phase), len(t)).
        f = window_overlap_fraction(
            t[None, :], exptime, period, dur, phase[:, None] + min_t
        )

        # Weighted least squares fit of ``y = y_out - depth * f`` for every
        # trial phase.
        s_f = f @ ivar
        s_ff = (f * f) @ ivar
        s_fy = f @ ivar_y
        denom = s_w * s_ff - s_f * s_f
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = (s_f * s_y - s_w * s_fy) / denom
            depth_err = np.sqrt(s_w / denom)
        snr = depth / depth_err

        # Log likelihood of the box model relative to the constant
        # out-of-transit model.
        loglike = 0.5 * depth * depth * s_ff

        # Choose which objective should be used for the optimization.
        if use_likelihood:
            objective = loglike
        else:
            objective = snr
        objective = np.where((denom > 0.0) & (depth > 0.0), objective, -np.inf)

        # If this model is better than any before, keep it.
        i_best = np.argmax(objective)
        if objective[i_best] > best[0]:
            best = (
                objective[i_best],
                (
                    objective[i_best],
                    depth[i_best],
                    depth_err[i_best],
                    dur,
                    (phase[i_best] + min_t) % period,
                    snr[i_best],
                    loglike[i_best],
                ),
            )

    return best[1]




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


def _apply(f, period):
    return tuple(map(np.array, zip(*map(f, period))))
