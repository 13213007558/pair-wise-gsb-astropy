# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np


__all__ = ["single_transit_fraction", "transit_fraction"]


def transit_fraction(t, exposure_time, period, duration, transit_time):
    """Fraction of each exposure window overlapping periodic transit windows."""
    t = np.asarray(t, dtype=np.float64)
    exposure_time = np.asarray(exposure_time, dtype=np.float64)
    t, exposure_time = np.broadcast_arrays(t, exposure_time)

    fraction = (
        np.abs((t - transit_time + 0.5 * period) % period - 0.5 * period)
        < 0.5 * duration
    ).astype(np.float64)
    nonzero = exposure_time > 0.0
    if not np.any(nonzero):
        return fraction

    exposure = exposure_time[nonzero]
    full_cycles = np.floor(exposure / period)
    residual = exposure - full_cycles * period
    start = np.mod(t[nonzero] - transit_time - 0.5 * exposure, period)
    end = start + residual
    half_duration = 0.5 * duration

    overlap_start = np.minimum(end, period)
    overlap_0 = np.maximum(
        0.0,
        np.minimum(overlap_start, half_duration) - np.maximum(start, 0.0),
    )
    overlap_period = np.maximum(
        0.0,
        np.minimum(end, period + half_duration)
        - np.maximum(start, period - half_duration),
    )
    overlap_wrapped = np.maximum(
        0.0, end - np.maximum(start, 2.0 * period - half_duration)
    )
    fraction[nonzero] = (
        full_cycles * duration
        + overlap_0
        + overlap_period
        + overlap_wrapped
    ) / exposure
    return np.clip(fraction, 0.0, 1.0)


def single_transit_fraction(
    t, exposure_time, period, duration, transit_time, cycle
):
    """Fraction of each exposure overlapping one transit in a periodic sequence."""
    t = np.asarray(t, dtype=np.float64)
    exposure_time = np.asarray(exposure_time, dtype=np.float64)
    instant = exposure_time == 0.0
    center = transit_time + cycle * period
    instant_fraction = (np.abs(t - center) < 0.5 * duration).astype(np.float64)

    start = t - 0.5 * exposure_time
    end = t + 0.5 * exposure_time
    transit_start = transit_time + cycle * period - 0.5 * duration
    transit_end = transit_start + duration

    overlap = np.maximum(
        0.0, np.minimum(end, transit_end) - np.maximum(start, transit_start)
    )
    fraction = np.divide(
        overlap,
        exposure_time,
        out=np.zeros_like(overlap),
        where=~instant,
    )
    return np.where(instant, instant_fraction, fraction)
