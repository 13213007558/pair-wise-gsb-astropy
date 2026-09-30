# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from astropy import units as u
from astropy.tests.helper import assert_quantity_allclose
from astropy.time import Time, TimeDelta
from astropy.timeseries.periodograms.bls import BoxLeastSquares
from astropy.timeseries.periodograms.lombscargle.core import has_units


def assert_allclose_blsresults(blsresult, other, **kwargs):
    """Assert that another BoxLeastSquaresResults object is consistent

    This method loops over all attributes and compares the values using
    :func:`~astropy.tests.helper.assert_quantity_allclose` function.

    Parameters
    ----------
    other : BoxLeastSquaresResults
        The other results object to compare.

    """
    __tracebackhide__ = True
    for k, v in blsresult.items():
        if k not in other:
            raise AssertionError(f"missing key '{k}'")
        if k == "objective":
            assert v == other[k], (
                f"Mismatched objectives. Expected '{v}', got '{other[k]}'"
            )
            continue
        assert_quantity_allclose(v, other[k], **kwargs)


# NOTE: PR 10644 replaced deprecated usage of RandomState but could not
# find a new seed that did not cause test failure, resorted to hardcoding.
@pytest.fixture
def data():
    rand = np.random.default_rng(454)
    t = rand.uniform(0, 10, 500)

    y = np.ones_like(t)
    dy = rand.uniform(0.005, 0.01, len(t))

    period = 2.0
    transit_time = 0.5
    duration = 0.16
    depth = 0.2
    m = (
        np.abs((t - transit_time + 0.5 * period) % period - 0.5 * period)
        < 0.5 * duration
    )
    y[m] = 1.0 - depth
    y += dy * rand.standard_normal(len(t))
    return (
        t,
        y,
        dy,
        dict(period=period, transit_time=transit_time, duration=duration, depth=depth),
    )


def test_32bit_bug():
    rand = np.random.default_rng(42)
    t = rand.uniform(0, 10, 500)
    y = np.ones_like(t)
    y[np.abs((t + 1.0) % 2.0 - 1) < 0.08] = 1.0 - 0.1
    y += 0.01 * rand.standard_normal(len(t))

    model = BoxLeastSquares(t, y)
    results = model.autopower(0.16)
    assert_allclose(results.period[np.argmax(results.power)], 2.000412388152837)
    periods = np.linspace(1.9, 2.1, 5)
    results = model.power(periods, 0.16)
    assert_allclose(
        results.power,
        [0.01723948, 0.0643028, 0.1338783, 0.09428816, 0.03577543],
        rtol=1.1e-7,
    )


@pytest.mark.parametrize("objective", ["likelihood", "snr"])
def test_correct_model(data, objective):
    t, y, dy, params = data
    model = BoxLeastSquares(t, y, dy)
    periods = np.exp(
        np.linspace(
            np.log(params["period"]) - 0.1, np.log(params["period"]) + 0.1, 1000
        )
    )
    results = model.power(periods, params["duration"], objective=objective)
    ind = np.argmax(results.power)
    for k, v in params.items():
        assert_allclose(results[k][ind], v, atol=0.01)
    chi = (results.depth[ind] - params["depth"]) / results.depth_err[ind]
    assert np.abs(chi) < 1


@pytest.mark.parametrize("objective", ["likelihood", "snr"])
@pytest.mark.parametrize("offset", [False, True])
def test_fast_method(data, objective, offset):
    t, y, dy, params = data
    if offset:
        t = t - params["transit_time"] + params["period"]
    model = BoxLeastSquares(t, y, dy)
    periods = np.exp(
        np.linspace(np.log(params["period"]) - 1, np.log(params["period"]) + 1, 10)
    )
    durations = params["duration"]
    results = model.power(periods, durations, objective=objective)

    assert_allclose_blsresults(
        results, model.power(periods, durations, method="slow", objective=objective)
    )


def test_input_units(data):
    t, y, dy, params = data

    t_unit = u.day
    y_unit = u.mag

    with pytest.raises(u.UnitConversionError):
        BoxLeastSquares(t * t_unit, y * y_unit, dy * u.one)
    with pytest.raises(u.UnitConversionError):
        BoxLeastSquares(t * t_unit, y * u.one, dy * y_unit)
    with pytest.raises(u.UnitConversionError):
        BoxLeastSquares(t * t_unit, y, dy * y_unit)
    model = BoxLeastSquares(t * t_unit, y * u.one, dy)
    assert model.dy.unit == model.y.unit
    model = BoxLeastSquares(t * t_unit, y * y_unit, dy)
    assert model.dy.unit == model.y.unit
    model = BoxLeastSquares(t * t_unit, y * y_unit)
    assert model.dy is None


def test_period_units(data):
    t, y, dy, params = data
    t_unit = u.day
    y_unit = u.mag
    model = BoxLeastSquares(t * t_unit, y * y_unit, dy)

    p = model.autoperiod(params["duration"])
    assert p.unit == t_unit
    p = model.autoperiod(params["duration"] * 24 * u.hour)
    assert p.unit == t_unit
    with pytest.raises(u.UnitConversionError):
        model.autoperiod(params["duration"] * u.mag)

    p = model.autoperiod(params["duration"], minimum_period=0.5)
    assert p.unit == t_unit
    with pytest.raises(u.UnitConversionError):
        p = model.autoperiod(params["duration"], minimum_period=0.5 * u.mag)

    p = model.autoperiod(params["duration"], maximum_period=0.5)
    assert p.unit == t_unit
    with pytest.raises(u.UnitConversionError):
        p = model.autoperiod(params["duration"], maximum_period=0.5 * u.mag)

    p = model.autoperiod(params["duration"], minimum_period=0.5, maximum_period=1.5)
    p2 = model.autoperiod(params["duration"], maximum_period=0.5, minimum_period=1.5)
    assert_quantity_allclose(p, p2)


@pytest.mark.parametrize("method", ["fast", "slow"])
@pytest.mark.parametrize("with_err", [True, False])
@pytest.mark.parametrize("t_unit", [None, u.day])
@pytest.mark.parametrize("y_unit", [None, u.mag])
@pytest.mark.parametrize("objective", ["likelihood", "snr"])
def test_results_units(data, method, with_err, t_unit, y_unit, objective):
    t, y, dy, params = data

    periods = np.linspace(params["period"] - 1.0, params["period"] + 1.0, 3)

    if t_unit is not None:
        t = t * t_unit
    if y_unit is not None:
        y = y * y_unit
        dy = dy * y_unit
    if not with_err:
        dy = None

    model = BoxLeastSquares(t, y, dy)
    results = model.power(
        periods, params["duration"], method=method, objective=objective
    )

    if t_unit is None:
        assert not has_units(results.period)
        assert not has_units(results.duration)
        assert not has_units(results.transit_time)
    else:
        assert results.period.unit == t_unit
        assert results.duration.unit == t_unit
        assert results.transit_time.unit == t_unit

    if y_unit is None:
        assert not has_units(results.power)
        assert not has_units(results.depth)
        assert not has_units(results.depth_err)
        assert not has_units(results.depth_snr)
        assert not has_units(results.log_likelihood)
    else:
        assert results.depth.unit == y_unit
        assert results.depth_err.unit == y_unit
        assert results.depth_snr.unit == u.one

        if dy is None:
            assert results.log_likelihood.unit == y_unit * y_unit
            if objective == "snr":
                assert results.power.unit == u.one
            else:
                assert results.power.unit == y_unit * y_unit
        else:
            assert results.log_likelihood.unit == u.one
            assert results.power.unit == u.one


def test_autopower(data):
    t, y, dy, params = data
    duration = params["duration"] + np.linspace(-0.1, 0.1, 3)

    model = BoxLeastSquares(t, y, dy)
    period = model.autoperiod(duration)
    results1 = model.power(period, duration)
    results2 = model.autopower(duration)

    assert_allclose_blsresults(results1, results2)


@pytest.mark.parametrize("with_units", [True, False])
def test_model(data, with_units):
    t, y, dy, params = data

    # Compute the model using linear regression
    A = np.zeros((len(t), 2))
    p = params["period"]
    dt = np.abs((t - params["transit_time"] + 0.5 * p) % p - 0.5 * p)
    m_in = dt < 0.5 * params["duration"]
    A[~m_in, 0] = 1.0
    A[m_in, 1] = 1.0
    w = np.linalg.solve(np.dot(A.T, A / dy[:, None] ** 2), np.dot(A.T, y / dy**2))
    model_true = np.dot(A, w)

    if with_units:
        t = t * u.day
        y = y * u.mag
        dy = dy * u.mag
        model_true = model_true * u.mag

    # Compute the model using the periodogram
    pgram = BoxLeastSquares(t, y, dy)
    model = pgram.model(t, p, params["duration"], params["transit_time"])

    # Make sure that the transit mask is consistent with the model
    transit_mask = pgram.transit_mask(t, p, params["duration"], params["transit_time"])
    transit_mask0 = (model - model.max()) < 0.0
    assert_allclose(transit_mask, transit_mask0)

    assert_quantity_allclose(model, model_true)


@pytest.mark.parametrize("shape", [(1,), (2,), (3,), (2, 3)])
def test_shapes(data, shape):
    t, y, dy, params = data
    duration = params["duration"]
    model = BoxLeastSquares(t, y, dy)

    period = np.empty(shape)
    period.flat = np.linspace(params["period"] - 1, params["period"] + 1, period.size)
    if len(period.shape) > 1:
        with pytest.raises(ValueError):
            results = model.power(period, duration)
    else:
        results = model.power(period, duration)
        for k, v in results.items():
            if k == "objective":
                continue
            assert v.shape == shape


@pytest.mark.parametrize("with_units", [True, False])
@pytest.mark.parametrize("with_err", [True, False])
def test_compute_stats(data, with_units, with_err):
    t, y, dy, params = data

    y_unit = 1
    if with_units:
        y_unit = u.mag
        t = t * u.day
        y = y * u.mag
        dy = dy * u.mag
        params["period"] = params["period"] * u.day
        params["duration"] = params["duration"] * u.day
        params["transit_time"] = params["transit_time"] * u.day
        params["depth"] = params["depth"] * u.mag
    if not with_err:
        dy = None

    model = BoxLeastSquares(t, y, dy)
    results = model.power(params["period"], params["duration"], oversample=1000)
    stats = model.compute_stats(
        params["period"], params["duration"], params["transit_time"]
    )

    # Test the calculated transit times
    tt = params["period"] * np.arange(int(t.max() / params["period"]) + 1)
    tt += params["transit_time"]
    assert_quantity_allclose(tt, stats["transit_times"])

    # Test that the other parameters are consistent with the periodogram
    assert_allclose(stats["per_transit_count"], [5, 11, 9, 8, 17])
    assert_quantity_allclose(
        np.sum(stats["per_transit_log_likelihood"]), results["log_likelihood"]
    )
    assert_quantity_allclose(stats["depth"][0], results["depth"])

    # Check the half period result
    results_half = model.power(
        0.5 * params["period"], params["duration"], oversample=1000
    )
    assert_quantity_allclose(stats["depth_half"][0], results_half["depth"])

    # Skip the uncertainty tests when the input errors are None
    if not with_err:
        assert_quantity_allclose(
            stats["harmonic_amplitude"], 0.039298120413786705 * y_unit
        )
        assert_quantity_allclose(
            stats["harmonic_delta_log_likelihood"],
            -0.7080424630782493 * y_unit * y_unit,
        )
        return

    assert_quantity_allclose(stats["harmonic_amplitude"], 0.04047660800393152 * y_unit)
    assert_quantity_allclose(
        stats["harmonic_delta_log_likelihood"], -14510.408959100823
    )

    assert_quantity_allclose(stats["depth"][1], results["depth_err"])
    assert_quantity_allclose(stats["depth_half"][1], results_half["depth_err"])
    for f, k in zip(
        (1.0, 1.0, 1.0, 0.0), ("depth", "depth_even", "depth_odd", "depth_phased")
    ):
        res = np.abs((stats[k][0] - f * params["depth"]) / stats[k][1])
        assert res < 1, f"f={f}, k={k}, res={res}"


def test_negative_times(data):
    t, y, dy, params = data
    mu = np.mean(t)
    duration = params["duration"] + np.linspace(-0.1, 0.1, 3)

    model1 = BoxLeastSquares(t, y, dy)
    results1 = model1.autopower(duration)

    # Compute the periodogram with offset (negative) times
    model2 = BoxLeastSquares(t - mu, y, dy)
    results2 = model2.autopower(duration)

    # Shift the transit times back into the unshifted coordinates
    results2.transit_time = (results2.transit_time + mu) % results2.period

    assert_allclose_blsresults(results1, results2)


@pytest.mark.parametrize("timedelta", [False, True])
def test_absolute_times(data, timedelta):
    # Make sure that we handle absolute times correctly. We also check that
    # TimeDelta works properly when timedelta is True.

    # The example data uses relative times
    t, y, dy, params = data

    # Add units
    t = t * u.day
    y = y * u.mag
    dy = dy * u.mag

    # We now construct a set of absolute times but keeping the rest the same.
    start = Time("2019-05-04T12:34:56")
    trel = TimeDelta(t) if timedelta else t
    t = trel + start

    # and we set up two instances of BoxLeastSquares, one with absolute and one
    # with relative times.
    bls1 = BoxLeastSquares(t, y, dy)
    bls2 = BoxLeastSquares(trel, y, dy)

    results1 = bls1.autopower(0.16 * u.day)
    results2 = bls2.autopower(0.16 * u.day)

    # All the results should match except transit time which should be
    # absolute instead of relative in the first case.

    for key in results1:
        if key == "transit_time":
            assert_quantity_allclose((results1[key] - start).to(u.day), results2[key])
        elif key == "objective":
            assert results1[key] == results2[key]
        else:
            assert_allclose(results1[key], results2[key])

    # Check that model evaluation works fine

    model1 = bls1.model(t, 0.2 * u.day, 0.05 * u.day, Time("2019-06-04T12:34:56"))
    model2 = bls2.model(trel, 0.2 * u.day, 0.05 * u.day, TimeDelta(1 * u.day))
    assert_quantity_allclose(model1, model2)

    # Check model validation
    MESSAGE = (
        r"{} was provided as {} time but the BoxLeastSquares class was initialized with"
        r" {} times\."
    )

    with pytest.raises(
        TypeError, match=MESSAGE.format("transit_time", "a relative", "absolute")
    ):
        bls1.model(t, 0.2 * u.day, 0.05 * u.day, 1 * u.day)

    with pytest.raises(
        TypeError, match=MESSAGE.format("t_model", "a relative", "absolute")
    ):
        bls1.model(trel, 0.2 * u.day, 0.05 * u.day, Time("2019-06-04T12:34:56"))

    with pytest.raises(
        TypeError, match=MESSAGE.format("transit_time", "an absolute", "relative")
    ):
        bls2.model(trel, 0.2 * u.day, 0.05 * u.day, Time("2019-06-04T12:34:56"))

    with pytest.raises(
        TypeError, match=MESSAGE.format("t_model", "an absolute", "relative")
    ):
        bls2.model(t, 0.2 * u.day, 0.05 * u.day, 1 * u.day)

    # Check compute_stats

    stats1 = bls1.compute_stats(0.2 * u.day, 0.05 * u.day, Time("2019-06-04T12:34:56"))
    stats2 = bls2.compute_stats(0.2 * u.day, 0.05 * u.day, 1 * u.day)

    for key in stats1:
        if key == "transit_times":
            assert_quantity_allclose(
                (stats1[key] - start).to(u.day), stats2[key], atol=1e-10 * u.day
            )
        elif key.startswith("depth"):
            for value1, value2 in zip(stats1[key], stats2[key]):
                assert_quantity_allclose(value1, value2)
        else:
            assert_allclose(stats1[key], stats2[key])

    # Check compute_stats validation
    MESSAGE = (
        r"{} was provided as {} time but the BoxLeastSquares class was"
        r" initialized with {} times\."
    )

    with pytest.raises(
        TypeError, match=MESSAGE.format("transit_time", "a relative", "absolute")
    ):
        bls1.compute_stats(0.2 * u.day, 0.05 * u.day, 1 * u.day)

    with pytest.raises(
        TypeError, match=MESSAGE.format("transit_time", "an absolute", "relative")
    ):
        bls2.compute_stats(0.2 * u.day, 0.05 * u.day, Time("2019-06-04T12:34:56"))

    # Check transit_mask

    mask1 = bls1.transit_mask(t, 0.2 * u.day, 0.05 * u.day, Time("2019-06-04T12:34:56"))
    mask2 = bls2.transit_mask(trel, 0.2 * u.day, 0.05 * u.day, 1 * u.day)

    assert_equal(mask1, mask2)

    # Check transit_mask validation

    with pytest.raises(
        TypeError, match=MESSAGE.format("transit_time", "a relative", "absolute")
    ):
        bls1.transit_mask(t, 0.2 * u.day, 0.05 * u.day, 1 * u.day)

    with pytest.raises(TypeError, match=MESSAGE.format("t", "a relative", "absolute")):
        bls1.transit_mask(trel, 0.2 * u.day, 0.05 * u.day, Time("2019-06-04T12:34:56"))

    with pytest.raises(
        TypeError, match=MESSAGE.format("transit_time", "an absolute", "relative")
    ):
        bls2.transit_mask(trel, 0.2 * u.day, 0.05 * u.day, Time("2019-06-04T12:34:56"))

    with pytest.raises(TypeError, match=MESSAGE.format("t", "an absolute", "relative")):
        bls2.transit_mask(t, 0.2 * u.day, 0.05 * u.day, 1 * u.day)


def test_transit_time_in_range(data):
    t, y, dy, params = data

    t_ref = 10230.0
    t2 = t + t_ref
    bls1 = BoxLeastSquares(t, y, dy)
    bls2 = BoxLeastSquares(t2, y, dy)

    results1 = bls1.autopower(0.16)
    results2 = bls2.autopower(0.16)

    assert np.allclose(results1.transit_time, results2.transit_time - t_ref)
    assert np.all(results1.transit_time >= t.min())
    assert np.all(results1.transit_time <= t.max())
    assert np.all(results2.transit_time >= t2.min())
    assert np.all(results2.transit_time <= t2.max())


def test_window_overlap_fraction():
    from astropy.timeseries.periodograms.bls.methods import window_overlap_fraction

    # Instantaneous samples of the box light curve
    f = window_overlap_fraction([0.5, 0.9], [0.0, 0.0], 2.0, 0.2, 0.5)
    assert_allclose(f, [1.0, 0.0])

    # Window fully inside the transit
    f = window_overlap_fraction([0.5], [0.1], 2.0, 0.4, 0.5)
    assert_allclose(f, [1.0])

    # Window straddling the transit edge
    f = window_overlap_fraction([0.55], [0.2], 2.0, 0.2, 0.5)
    assert_allclose(f, [0.75])

    # Window outside the transit
    f = window_overlap_fraction([0.75], [0.2], 2.0, 0.2, 0.5)
    assert_allclose(f, [0.0])

    # Transit window wrapped across the phase zero point
    f = window_overlap_fraction([0.05], [0.2], 2.0, 0.2, 0.05)
    assert_allclose(f, [1.0])

    # Exposure longer than one period averages to duration / period
    f = window_overlap_fraction([1.0], [6.0], 2.0, 0.2, 0.5)
    assert_allclose(f, [0.1])

    # Mixed zero and finite exposure times
    f = window_overlap_fraction([0.5, 0.5], [0.0, 0.4], 2.0, 0.2, 0.5)
    assert_allclose(f, [1.0, 0.5])


def test_exptime_validation(data):
    t, y, dy, params = data

    # Negative exposure times are not allowed
    with pytest.raises(ValueError, match="non-negative"):
        BoxLeastSquares(t, y, dy, exptime=-np.ones_like(t))

    # Scalars are broadcast to the shape of t
    model = BoxLeastSquares(t, y, dy, exptime=0.1)
    assert model.exptime.shape == t.shape

    # Units must be consistent with t
    with pytest.raises(u.UnitConversionError):
        BoxLeastSquares(t * u.day, y, dy, exptime=0.1 * u.mag)
    model = BoxLeastSquares(t * u.day, y, dy, exptime=2.4 * u.hour)
    assert model.exptime.unit == u.day
    assert_quantity_allclose(model.exptime, 0.1 * np.ones_like(t) * u.day)


@pytest.mark.parametrize("objective", ["likelihood", "snr"])
def test_exptime_zero_equivalence(data, objective):
    """Zero exposure times must degrade to the classical results."""
    t, y, dy, params = data
    model_ref = BoxLeastSquares(t, y, dy)
    model_zero = BoxLeastSquares(t, y, dy, exptime=np.zeros_like(t))

    periods = np.linspace(params["period"] - 0.1, params["period"] + 0.1, 5)
    results_ref = model_ref.power(
        periods, params["duration"], method="slow", objective=objective
    )
    results_zero = model_zero.power(periods, params["duration"], objective=objective)
    assert_allclose_blsresults(results_ref, results_zero, rtol=1e-10)

    y_ref = model_ref.model(
        t, params["period"], params["duration"], params["transit_time"]
    )
    y_zero = model_zero.model(
        t, params["period"], params["duration"], params["transit_time"]
    )
    assert_allclose(y_ref, y_zero, rtol=1e-10)

    mask_ref = model_ref.transit_mask(
        t, params["period"], params["duration"], params["transit_time"]
    )
    mask_zero = model_zero.transit_mask(
        t,
        params["period"],
        params["duration"],
        params["transit_time"],
        exptime=np.zeros_like(t),
    )
    assert_equal(mask_ref, mask_zero)

    stats_ref = model_ref.compute_stats(
        params["period"], params["duration"], params["transit_time"]
    )
    stats_zero = model_zero.compute_stats(
        params["period"], params["duration"], params["transit_time"]
    )
    for key in (
        "depth",
        "depth_odd",
        "depth_even",
        "depth_half",
        "depth_phased",
        "per_transit_count",
        "per_transit_log_likelihood",
        "transit_times",
        "harmonic_amplitude",
        "harmonic_delta_log_likelihood",
    ):
        assert_allclose(stats_ref[key], stats_zero[key], rtol=1e-8)


def test_exptime_depth_recovery():
    """The exptime-aware mode recovers the depth of a smeared transit."""
    from astropy.timeseries.periodograms.bls.methods import window_overlap_fraction

    rng = np.random.default_rng(7)
    period, duration, depth, t0 = 3.5, 0.12, 0.02, 1.1
    exptime = 0.5  # much longer than the transit duration
    t = np.arange(exptime / 2, 70.0, exptime)
    dy = 0.001 * np.ones_like(t)
    et = np.full_like(t, exptime)
    y = 1.0 - depth * window_overlap_fraction(t, et, period, duration, t0)
    y += dy * rng.standard_normal(len(t))

    # The classical mode only recovers the smeared depth
    model_c = BoxLeastSquares(t, y, dy)
    results_c = model_c.power([period], [duration], oversample=50)
    k = np.argmax(results_c.power)
    smeared = depth * duration / exptime
    assert abs(results_c.depth[k] - smeared) < 3 * results_c.depth_err[k]

    # The exposure-aware mode recovers the injected physical depth
    model = BoxLeastSquares(t, y, dy, exptime=exptime)
    results = model.power([period], [duration], oversample=50)
    i = np.argmax(results.power)
    assert abs(results.depth[i] - depth) < 3 * results.depth_err[i]

    # compute_stats, model, and transit_mask use the same window semantics
    stats = model.compute_stats(period, duration, results.transit_time[i])
    assert abs(stats["depth"][0] - depth) < 3 * stats["depth"][1]

    y_model = model.model(t, period, duration, results.transit_time[i], exptime=exptime)
    assert np.std(y - y_model) < 1.5 * dy[0]

    mask = model.transit_mask(
        t, period, duration, results.transit_time[i], exptime=exptime
    )
    f_expected = window_overlap_fraction(
        t, et, period, duration, results.transit_time[i]
    )
    assert np.all(mask == (f_expected > 0))


def test_exptime_noiseless_exact():
    """Without noise the exposure-aware depth estimate is exact."""
    from astropy.timeseries.periodograms.bls.methods import window_overlap_fraction

    period, duration, depth, t0 = 3.5, 0.12, 0.02, 1.1
    exptime = 0.5
    t = np.arange(exptime / 2, 70.0, exptime)
    et = np.full_like(t, exptime)
    y = 1.0 - depth * window_overlap_fraction(t, et, period, duration, t0)

    model = BoxLeastSquares(t, y, exptime=exptime)
    stats = model.compute_stats(period, duration, t0)
    assert_allclose(stats["depth"][0], depth, rtol=1e-10)

    y_model = model.model(t, period, duration, t0, exptime=exptime)
    assert_allclose(y_model, y, atol=1e-10)
