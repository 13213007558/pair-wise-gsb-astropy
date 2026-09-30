# Licensed under a 3-clause BSD style license - see LICENSE.rst

__all__ = ["BoxLeastSquares", "BoxLeastSquaresResults"]

import numpy as np

from astropy import units
from astropy import units as u
from astropy.time import Time, TimeDelta
from astropy.timeseries.periodograms.base import BasePeriodogram
from astropy.timeseries.periodograms.lombscargle.core import has_units, strip_units

from . import methods
from .window import single_transit_fraction, transit_fraction


def validate_unit_consistency(reference_object, input_object):
    if has_units(reference_object):
        input_object = units.Quantity(input_object, unit=reference_object.unit)
    else:
        if has_units(input_object):
            input_object = units.Quantity(input_object, unit=units.one)
            input_object = input_object.value
    return input_object


class BoxLeastSquares(BasePeriodogram):
    """Compute the box least squares periodogram.

    This method is a commonly used tool for discovering transiting exoplanets
    or eclipsing binaries in photometric time series datasets. This
    implementation is based on the "box least squares (BLS)" method described
    in [1]_ and [2]_.

    Parameters
    ----------
    t : array-like, `~astropy.units.Quantity`, `~astropy.time.Time`, or `~astropy.time.TimeDelta`
        Sequence of observation times.
    y : array-like or `~astropy.units.Quantity`
        Sequence of observations associated with times ``t``.
    dy : float, array-like, or `~astropy.units.Quantity`, optional
        Error or sequence of observational errors associated with times ``t``.
    exposure_time : float, array-like, or `~astropy.units.Quantity` ['time'], optional
        Duration of each observation exposure, centered on the corresponding
        value of ``t``. When provided, model values are averaged over each
        exposure window. Exposures may cross a phase boundary or span more
        than one period. A value of zero, the default, treats observations as
        instantaneous.

    Examples
    --------
    Generate noisy data with a transit:

    >>> rand = np.random.default_rng(42)
    >>> t = rand.uniform(0, 10, 500)
    >>> y = np.ones_like(t)
    >>> y[np.abs((t + 1.0)%2.0-1)<0.08] = 1.0 - 0.1
    >>> y += 0.01 * rand.standard_normal(len(t))

    Compute the transit periodogram on a heuristically determined period grid
    and find the period with maximum power:

    >>> model = BoxLeastSquares(t, y)
    >>> results = model.autopower(0.16)
    >>> results.period[np.argmax(results.power)]  # doctest: +FLOAT_CMP
    np.float64(2.000412388152837)

    Compute the periodogram on a user-specified period grid:

    >>> periods = np.linspace(1.9, 2.1, 5)
    >>> results = model.power(periods, 0.16)
    >>> results.power  # doctest: +FLOAT_CMP
    array([0.01723948, 0.0643028 , 0.1338783 , 0.09428816, 0.03577543])

    If the inputs are AstroPy Quantities with units, the units will be
    validated and the outputs will also be Quantities with appropriate units:

    >>> from astropy import units as u
    >>> t = t * u.day
    >>> y = y * u.dimensionless_unscaled
    >>> model = BoxLeastSquares(t, y)
    >>> results = model.autopower(0.16 * u.day)
    >>> results.period.unit
    Unit("d")
    >>> results.power.unit
    Unit(dimensionless)

    References
    ----------
    .. [1] Kovacs, Zucker, & Mazeh (2002), A&A, 391, 369
        (arXiv:astro-ph/0206099)
    .. [2] Hartman & Bakos (2016), Astronomy & Computing, 17, 1
        (arXiv:1605.06811)

    """

    def __init__(self, t, y, dy=None, exposure_time=None):
        # If t is a TimeDelta, convert it to a quantity. The units we convert
        # to don't really matter since the user gets a Quantity back at the end
        # so can convert to any units they like.
        if isinstance(t, TimeDelta):
            t = t.to("day")

        # We want to expose self.t as being the times the user passed in, but
        # if the times are absolute, we need to convert them to relative times
        # internally, so we use self._trel and self._tstart for this.

        self.t = t

        if isinstance(self.t, (Time, TimeDelta)):
            self._tstart = self.t[0]
            trel = (self.t - self._tstart).to(u.day)
        else:
            self._tstart = None
            trel = self.t

        self.exposure_time = exposure_time
        self._trel, self.y, self.dy, self.exposure_time = self._validate_inputs(
            trel, y, dy, exposure_time
        )

    def autoperiod(
        self,
        duration,
        minimum_period=None,
        maximum_period=None,
        minimum_n_transit=3,
        frequency_factor=1.0,
    ):
        """Determine a suitable grid of periods.

        This method uses a set of heuristics to select a conservative period
        grid that is uniform in frequency. This grid might be too fine for
        some user's needs depending on the precision requirements or the
        sampling of the data. The grid can be made coarser by increasing
        ``frequency_factor``.

        Parameters
        ----------
        duration : float, array-like, or `~astropy.units.Quantity` ['time']
            The set of durations that will be considered.
        minimum_period, maximum_period : float or `~astropy.units.Quantity` ['time'], optional
            The minimum/maximum periods to search. If not provided, these will
            be computed as described in the notes below.
        minimum_n_transit : int, optional
            If ``maximum_period`` is not provided, this is used to compute the
            maximum period to search by asserting that any systems with at
            least ``minimum_n_transits`` will be within the range of searched
            periods. Note that this is not the same as requiring that
            ``minimum_n_transits`` be required for detection. The default
            value is ``3``.
        frequency_factor : float, optional
            A factor to control the frequency spacing as described in the
            notes below. The default value is ``1.0``.

        Returns
        -------
        period : array-like or `~astropy.units.Quantity` ['time']
            The set of periods computed using these heuristics with the same
            units as ``t``.

        Notes
        -----
        The default minimum period is chosen to be twice the maximum duration
        because there won't be much sensitivity to periods shorter than that.

        The default maximum period is computed as

        .. code-block:: python

            maximum_period = (max(t) - min(t)) / minimum_n_transits

        ensuring that any systems with at least ``minimum_n_transits`` are
        within the range of searched periods.

        The frequency spacing is given by

        .. code-block:: python

            df = frequency_factor * min(duration) / (max(t) - min(t))**2

        so the grid can be made finer by decreasing ``frequency_factor`` or
        coarser by increasing ``frequency_factor``.

        """
        duration = self._validate_duration(duration)
        baseline = strip_units(self._trel.max() - self._trel.min())
        min_duration = strip_units(np.min(duration))

        # Estimate the required frequency spacing
        # Because of the sparsity of a transit, this must be much finer than
        # the frequency resolution for a sinusoidal fit. For a sinusoidal fit,
        # df would be 1/baseline (see LombScargle), but here this should be
        # scaled proportionally to the duration in units of baseline.
        df = frequency_factor * min_duration / baseline**2

        # If a minimum period is not provided, choose one that is twice the
        # maximum duration because we won't be sensitive to any periods
        # shorter than that.
        if minimum_period is None:
            minimum_period = 2.0 * strip_units(np.max(duration))
        else:
            minimum_period = validate_unit_consistency(self._trel, minimum_period)
            minimum_period = strip_units(minimum_period)

        # If no maximum period is provided, choose one by requiring that
        # all signals with at least minimum_n_transit should be detectable.
        if maximum_period is None:
            if minimum_n_transit <= 1:
                raise ValueError("minimum_n_transit must be greater than 1")
            maximum_period = baseline / (minimum_n_transit - 1)
        else:
            maximum_period = validate_unit_consistency(self._trel, maximum_period)
            maximum_period = strip_units(maximum_period)

        if maximum_period < minimum_period:
            minimum_period, maximum_period = maximum_period, minimum_period
        if minimum_period <= 0.0:
            raise ValueError("minimum_period must be positive")

        # Convert bounds to frequency
        minimum_frequency = 1.0 / strip_units(maximum_period)
        maximum_frequency = 1.0 / strip_units(minimum_period)

        # Compute the number of frequencies and the frequency grid
        nf = 1 + int(np.round((maximum_frequency - minimum_frequency) / df))
        return 1.0 / (maximum_frequency - df * np.arange(nf)) * self._t_unit()

    def autopower(
        self,
        duration,
        objective=None,
        method=None,
        oversample=10,
        minimum_n_transit=3,
        minimum_period=None,
        maximum_period=None,
        frequency_factor=1.0,
    ):
        """Compute the periodogram at set of heuristically determined periods.

        This method calls :func:`BoxLeastSquares.autoperiod` to determine
        the period grid and then :func:`BoxLeastSquares.power` to compute
        the periodogram. See those methods for documentation of the arguments.

        """
        period = self.autoperiod(
            duration,
            minimum_n_transit=minimum_n_transit,
            minimum_period=minimum_period,
            maximum_period=maximum_period,
            frequency_factor=frequency_factor,
        )
        return self.power(
            period, duration, objective=objective, method=method, oversample=oversample
        )

    def power(self, period, duration, objective=None, method=None, oversample=10):
        """Compute the periodogram for a set of periods.

        Parameters
        ----------
        period : array-like or `~astropy.units.Quantity` ['time']
            The periods where the power should be computed
        duration : float, array-like, or `~astropy.units.Quantity` ['time']
            The set of durations to test
        objective : {'likelihood', 'snr'}, optional
            The scalar that should be optimized to find the best fit phase,
            duration, and depth. This can be either ``'likelihood'`` (default)
            to optimize the log-likelihood of the model, or ``'snr'`` to
            optimize the signal-to-noise with which the transit depth is
            measured.
        method : {'fast', 'slow'}, optional
            The computational method used to compute the periodogram. This is
            mainly included for the purposes of testing and most users will
            want to use the optimized ``'fast'`` method (default) that is
            implemented in Cython for instantaneous observations. When
            ``exposure_time`` is nonzero, both methods evaluate the exact
            exposure-window average, with ``'slow'`` retained as a reference.
        oversample : int, optional
            The number of bins per duration that should be used. This sets the
            time resolution of the phase fit with larger values of
            ``oversample`` yielding a finer grid and higher computational cost.

        Returns
        -------
        results : BoxLeastSquaresResults
            The periodogram results as a :class:`BoxLeastSquaresResults`
            object.

        Raises
        ------
        ValueError
            If ``oversample`` is not an integer greater than 0 or if
            ``objective`` or ``method`` are not valid.

        """
        period, duration = self._validate_period_and_duration(period, duration)

        # Check for absurdities in the ``oversample`` choice
        try:
            oversample = int(oversample)
        except TypeError:
            raise ValueError(f"oversample must be an int, got {oversample}")
        if oversample < 1:
            raise ValueError("oversample must be greater than or equal to 1")

        # Select the periodogram objective
        if objective is None:
            objective = "likelihood"
        allowed_objectives = ["snr", "likelihood"]
        if objective not in allowed_objectives:
            raise ValueError(
                f"Unrecognized method '{objective}'\n"
                f"allowed methods are: {allowed_objectives}"
            )
        use_likelihood = objective == "likelihood"

        # Select the computational method
        if method is None:
            method = "fast"
        allowed_methods = ["fast", "slow"]
        if method not in allowed_methods:
            raise ValueError(
                f"Unrecognized method '{method}'\n"
                f"allowed methods are: {allowed_methods}"
            )

        # Format and check the input arrays
        t = np.ascontiguousarray(strip_units(self._trel), dtype=np.float64)
        t_ref = np.min(t)
        y = np.ascontiguousarray(strip_units(self.y), dtype=np.float64)
        if self.dy is None:
            ivar = np.ones_like(y)
        else:
            ivar = (
                1.0 / np.ascontiguousarray(strip_units(self.dy), dtype=np.float64) ** 2
            )

        # Make sure that the period and duration arrays are C-order
        period_fmt = np.ascontiguousarray(strip_units(period), dtype=np.float64)
        duration = np.ascontiguousarray(strip_units(duration), dtype=np.float64)
        exposure_time = np.ascontiguousarray(
            strip_units(self.exposure_time), dtype=np.float64
        )

        # Select the correct implementation for the chosen method
        if np.all(exposure_time == 0.0):
            bls = methods.bls_fast if method == "fast" else methods.bls_slow
        else:
            bls = methods.bls_integrated

        # Run the implementation
        if bls is methods.bls_integrated:
            results = bls(
                t - t_ref,
                y - np.median(y),
                ivar,
                exposure_time,
                period_fmt,
                duration,
                oversample,
                use_likelihood,
            )
        else:
            results = bls(
                t - t_ref,
                y - np.median(y),
                ivar,
                period_fmt,
                duration,
                oversample,
                use_likelihood,
            )

        return self._format_results(t_ref, objective, period, results)

    def _as_relative_time(self, name, times):
        """
        Convert the provided times (if absolute) to relative times using the
        current _tstart value. If the times provided are relative, they are
        returned without conversion (though we still do some checks).
        """
        if isinstance(times, TimeDelta):
            times = times.to("day")

        if self._tstart is None:
            if isinstance(times, Time):
                raise TypeError(
                    f"{name} was provided as an absolute time but "
                    "the BoxLeastSquares class was initialized "
                    "with relative times."
                )
        else:
            if isinstance(times, Time):
                times = (times - self._tstart).to(u.day)
            else:
                raise TypeError(
                    f"{name} was provided as a relative time but "
                    "the BoxLeastSquares class was initialized "
                    "with absolute times."
                )

        times = validate_unit_consistency(self._trel, times)

        return times

    def _as_absolute_time_if_needed(self, name, times):
        """
        Convert the provided times to absolute times using the current _tstart
        value, if needed.
        """
        if self._tstart is not None:
            # Some time formats/scales can't represent dates/times too far
            # off from the present, so we need to mask values offset by
            # more than 100,000 yr (the periodogram algorithm can return
            # transit times of e.g 1e300 for some periods).
            reset = np.abs(times.to_value(u.year)) > 100000
            times[reset] = 0
            times = self._tstart + times
            times[reset] = np.nan
        return times

    def _model_exposure_time(self, t_model, exposure_time):
        provided = exposure_time is not None
        if exposure_time is None:
            exposure_time = self.exposure_time
        if isinstance(exposure_time, TimeDelta):
            exposure_time = exposure_time.to(u.day)
        elif isinstance(exposure_time, Time):
            raise TypeError("exposure_time must be a duration, not an absolute time")
        exposure_time = validate_unit_consistency(self._trel, exposure_time)
        exposure_time = np.asarray(strip_units(exposure_time), dtype=np.float64)
        t_model = np.asarray(t_model)
        if exposure_time.ndim > 0 and exposure_time.shape != t_model.shape and provided:
            raise ValueError(
                "exposure_time must be scalar or have the same shape as t_model"
            )
        if exposure_time.ndim > 0 and exposure_time.shape != t_model.shape:
            exposure_time = np.zeros_like(t_model, dtype=np.float64)
        return np.broadcast_to(exposure_time, t_model.shape)

    def model(self, t_model, period, duration, transit_time, exposure_time=None):
        """Compute the transit model at the given period, duration, and phase.

        Parameters
        ----------
        t_model : array-like, `~astropy.units.Quantity`, or `~astropy.time.Time`
            Times at which to compute the model.
        period : float or `~astropy.units.Quantity` ['time']
            The period of the transits.
        duration : float or `~astropy.units.Quantity` ['time']
            The duration of the transit.
        transit_time : float or `~astropy.units.Quantity` or `~astropy.time.Time`
            The mid-transit time of a reference transit.
        exposure_time : float, array-like, or `~astropy.units.Quantity` ['time'], optional
            Exposure duration centered on each value of ``t_model``. The
            default is to use the exposure durations supplied when the
            periodogram was initialized.

        Returns
        -------
        y_model : array-like or `~astropy.units.Quantity`
            The exposure-averaged model evaluated at the times ``t_model``
            with units of ``y``.

        """
        period, duration = self._validate_period_and_duration(period, duration)

        transit_time = self._as_relative_time("transit_time", transit_time)
        t_model = strip_units(self._as_relative_time("t_model", t_model))

        period = float(strip_units(period[0]))
        duration = float(strip_units(duration[0]))
        transit_time = float(strip_units(transit_time))

        t = np.ascontiguousarray(strip_units(self._trel), dtype=np.float64)
        y = np.ascontiguousarray(strip_units(self.y), dtype=np.float64)
        if self.dy is None:
            ivar = np.ones_like(y)
        else:
            ivar = (
                1.0 / np.ascontiguousarray(strip_units(self.dy), dtype=np.float64) ** 2
            )

        train_fraction = transit_fraction(
            t,
            strip_units(self.exposure_time),
            period,
            duration,
            transit_time,
        )
        depth, y_out, _, _ = methods.weighted_box_stats(y, ivar, train_fraction)

        model_exposure = self._model_exposure_time(t_model, exposure_time)
        model_fraction = transit_fraction(
            t_model,
            model_exposure,
            period,
            duration,
            transit_time,
        )
        y_model = y_out - depth * model_fraction

        return y_model * self._y_unit()

    def compute_stats(self, period, duration, transit_time):
        """Compute descriptive statistics for a given transit model.

        These statistics are commonly used for vetting of transit candidates.

        Parameters
        ----------
        period : float or `~astropy.units.Quantity` ['time']
            The period of the transits.
        duration : float or `~astropy.units.Quantity` ['time']
            The duration of the transit.
        transit_time : float or `~astropy.units.Quantity` or `~astropy.time.Time`
            The mid-transit time of a reference transit.

        Returns
        -------
        stats : dict
            A dictionary containing several descriptive statistics:

            - ``depth``: The depth and uncertainty (as a tuple with two
                values) on the depth for the fiducial model.
            - ``depth_odd``: The depth and uncertainty on the depth for a
                model where the period is twice the fiducial period.
            - ``depth_even``: The depth and uncertainty on the depth for a
                model where the period is twice the fiducial period and the
                phase is offset by one orbital period.
            - ``depth_half``: The depth and uncertainty for a model with a
                period of half the fiducial period.
            - ``depth_phased``: The depth and uncertainty for a model with the
                fiducial period and the phase offset by half a period.
            - ``harmonic_amplitude``: The amplitude of the best fit sinusoidal
                model.
            - ``harmonic_delta_log_likelihood``: The difference in log
                likelihood between a sinusoidal model and the transit model.
                If ``harmonic_delta_log_likelihood`` is greater than zero, the
                sinusoidal model is preferred.
            - ``transit_times``: The mid-transit time for each transit in the
                baseline.
            - ``per_transit_count``: An array with a count of the number of
                data points in each unique transit included in the baseline.
            - ``per_transit_log_likelihood``: An array with the value of the
                log likelihood for each unique transit included in the
                baseline.

        """
        period, duration = self._validate_period_and_duration(period, duration)
        transit_time = self._as_relative_time("transit_time", transit_time)

        period = float(strip_units(period[0]))
        duration = float(strip_units(duration[0]))
        transit_time = float(strip_units(transit_time))

        t = np.ascontiguousarray(strip_units(self._trel), dtype=np.float64)
        y = np.ascontiguousarray(strip_units(self.y), dtype=np.float64)
        if self.dy is None:
            ivar = np.ones_like(y)
        else:
            ivar = (
                1.0 / np.ascontiguousarray(strip_units(self.dy), dtype=np.float64) ** 2
            )
        exposure_time = strip_units(self.exposure_time)

        def _box_stats(fraction):
            depth, baseline, depth_err, log_like = methods.weighted_box_stats(
                y, ivar, fraction
            )
            return depth, depth_err, baseline, log_like

        def _relative_depth(fraction, baseline, baseline_weight):
            group_weight = np.sum(ivar * fraction)
            if group_weight <= 0.0 or baseline_weight <= 0.0:
                return 0.0, np.inf
            group_flux = np.sum(ivar * fraction * y) / group_weight
            return baseline - group_flux, np.sqrt(
                1.0 / group_weight + 1.0 / baseline_weight
            )

        # Compute the depth of the fiducial model and the two models at twice
        # the period
        fraction = transit_fraction(t, exposure_time, period, duration, transit_time)
        fraction_odd = transit_fraction(
            t, exposure_time, 2 * period, duration, transit_time + period
        )
        fraction_even = transit_fraction(
            t, exposure_time, 2 * period, duration, transit_time
        )

        depth_value, depth_error, y_out, _ = _box_stats(fraction)
        depth = depth_value, depth_error
        baseline_weight = np.sum(ivar * (1.0 - fraction))
        depth_odd = _relative_depth(fraction_odd, y_out, baseline_weight)
        depth_even = _relative_depth(fraction_even, y_out, baseline_weight)

        # Compute the depth of the model at a phase of 0.5*period
        fraction_phase = transit_fraction(
            t, exposure_time, period, duration, transit_time + 0.5 * period
        )
        phase_baseline = (1.0 - fraction) * (1.0 - fraction_phase)
        phase_baseline_weight = np.sum(ivar * phase_baseline)
        phase_baseline_flux = (
            np.sum(ivar * phase_baseline * y) / phase_baseline_weight
            if phase_baseline_weight > 0.0
            else 0.0
        )
        depth_phase = _relative_depth(
            fraction_phase,
            phase_baseline_flux,
            phase_baseline_weight,
        )

        # Compute the depth of a model with a period of 0.5*period
        fraction_half = transit_fraction(
            t, exposure_time, 0.5 * period, duration, transit_time
        )
        depth_half = _box_stats(fraction_half)[:2]

        # Compute the overlap of each exposure with each transit
        exposure_broadcast = np.broadcast_to(exposure_time, t.shape)
        if np.all(exposure_broadcast == 0.0):
            transit_id = np.round((t[fraction > 0.0] - transit_time) / period).astype(
                int
            )
            k_min = int(np.min(transit_id))
            k_max = int(np.max(transit_id))
        else:
            k_min = int(
                np.floor(
                    np.min(
                        t - 0.5 * exposure_broadcast - transit_time - 0.5 * duration
                        / period
                    )
                )
            )
            k_max = int(
                np.ceil(
                    np.max(
                        t + 0.5 * exposure_broadcast - transit_time + 0.5 * duration
                        / period
                    )
                )
            )
        transit_ids = np.arange(k_min, k_max + 1)
        transit_times = period * transit_ids + transit_time
        counts = np.zeros(len(transit_ids))

        # Compute the per-transit log likelihood
        residual = y - y_out
        lls = np.zeros(len(transit_ids))
        for index, transit_id in enumerate(transit_ids):
            fraction_event = single_transit_fraction(
                t,
                exposure_broadcast,
                period,
                duration,
                transit_time,
                transit_id,
            )
            counts[index] = np.sum(fraction_event)
            event_fraction = np.divide(
                fraction_event,
                fraction,
                out=np.zeros_like(fraction),
                where=fraction > 0.0,
            )
            event_weight = event_fraction * ivar * fraction
            if np.all(exposure_broadcast == 0.0):
                lls[index] = np.sum(
                    event_weight
                    * 0.5
                    * ((y - y_out) ** 2 - (y - y_out + depth_value) ** 2)
                )
            else:
                lls[index] = np.sum(
                    event_weight
                    * (
                        depth_value * residual
                        - 0.5 * depth_value * depth_value * fraction
                    )
                )
        full_ll = -0.5 * np.sum(ivar * (y - (y_out - depth_value * fraction)) ** 2)

        # Compute the log likelihood of a sine model
        A = np.vstack(
            (
                np.sin(2 * np.pi * t / period),
                np.cos(2 * np.pi * t / period),
                np.ones_like(t),
            )
        ).T
        w = np.linalg.solve(np.dot(A.T, A * ivar[:, None]), np.dot(A.T, y * ivar))
        mod = np.dot(A, w)
        sin_ll = -0.5 * np.sum((y - mod) ** 2 * ivar)

        # Format the results
        y_unit = self._y_unit()
        ll_unit = 1
        if self.dy is None:
            ll_unit = y_unit * y_unit
        return dict(
            transit_times=self._as_absolute_time_if_needed(
                "transit_times", transit_times * self._t_unit()
            ),
            per_transit_count=counts,
            per_transit_log_likelihood=lls * ll_unit,
            depth=(depth[0] * y_unit, depth[1] * y_unit),
            depth_phased=(depth_phase[0] * y_unit, depth_phase[1] * y_unit),
            depth_half=(depth_half[0] * y_unit, depth_half[1] * y_unit),
            depth_odd=(depth_odd[0] * y_unit, depth_odd[1] * y_unit),
            depth_even=(depth_even[0] * y_unit, depth_even[1] * y_unit),
            harmonic_amplitude=np.sqrt(np.sum(w[:2] ** 2)) * y_unit,
            harmonic_delta_log_likelihood=(sin_ll - full_ll) * ll_unit,
        )

    def transit_mask(self, t, period, duration, transit_time, exposure_time=None):
        """Compute which data points are in transit for a given parameter set.

        Parameters
        ----------
        t : array-like or `~astropy.units.Quantity` ['time']
            Times where the mask should be evaluated.
        period : float or `~astropy.units.Quantity` ['time']
            The period of the transits.
        duration : float or `~astropy.units.Quantity` ['time']
            The duration of the transit.
        transit_time : float or `~astropy.units.Quantity` or `~astropy.time.Time`
            The mid-transit time of a reference transit.
        exposure_time : float, array-like, or `~astropy.units.Quantity` ['time'], optional
            Exposure duration centered on each value of ``t``.

        Returns
        -------
        transit_mask : array-like
            A boolean array where ``True`` indicates an exposure that overlaps
            a transit window and ``False`` indicates an out-of-transit point.

        """
        period, duration = self._validate_period_and_duration(period, duration)
        transit_time = self._as_relative_time("transit_time", transit_time)
        t = strip_units(self._as_relative_time("t", t))
        exposure_time = self._model_exposure_time(t, exposure_time)

        period = float(strip_units(period[0]))
        duration = float(strip_units(duration[0]))
        transit_time = float(strip_units(transit_time))

        return transit_fraction(t, exposure_time, period, duration, transit_time) > 0.0

    def _validate_inputs(self, t, y, dy, exposure_time=None):
        """Private method used to check the consistency of the inputs.

        Parameters
        ----------
        t : array-like, `~astropy.units.Quantity`, `~astropy.time.Time`, or `~astropy.time.TimeDelta`
            Sequence of observation times.
        y : array-like or `~astropy.units.Quantity`
            Sequence of observations associated with times t.
        dy : float, array-like, or `~astropy.units.Quantity`
            Error or sequence of observational errors associated with times t.
        exposure_time : float, array-like, or `~astropy.units.Quantity` ['time'], optional
            Duration of each observation exposure.

        Returns
        -------
        t, y, dy, exposure_time : array-like, `~astropy.units.Quantity`, or
            `~astropy.time.Time`
            The inputs with consistent shapes and units.

        Raises
        ------
        ValueError
            If the dimensions are incompatible or if the units of dy cannot be
            converted to the units of y.

        """
        # Validate shapes of inputs
        arrays = [t, y]
        if dy is not None:
            arrays.append(dy)
        if exposure_time is not None:
            if isinstance(exposure_time, Time):
                raise TypeError(
                    "exposure_time must be a duration, not an absolute time"
                )
            if isinstance(exposure_time, TimeDelta):
                exposure_time = exposure_time.to(u.day)
            exposure_time = validate_unit_consistency(t, exposure_time)
            arrays.append(exposure_time)
        arrays = np.broadcast_arrays(*arrays, subok=True)
        t, y = arrays[:2]
        if dy is not None:
            dy = arrays[2]
        if exposure_time is not None:
            exposure_time = arrays[-1]
        else:
            exposure_time = np.zeros_like(t, subok=True)
        if t.ndim != 1:
            raise ValueError("Inputs (t, y, dy, exposure_time) must be 1-dimensional")
        if np.any(np.asarray(strip_units(exposure_time)) < 0.0):
            raise ValueError("exposure_time must be non-negative")

        # validate units of inputs if any is a Quantity
        if dy is not None:
            dy = validate_unit_consistency(y, dy)

        return t, y, dy, exposure_time

    def _validate_duration(self, duration):
        """Private method used to check a set of test durations.

        Parameters
        ----------
        duration : float, array-like, or `~astropy.units.Quantity`
            The set of durations that will be considered.

        Returns
        -------
        duration : array-like or `~astropy.units.Quantity`
            The input reformatted with the correct shape and units.

        Raises
        ------
        ValueError
            If the units of duration cannot be converted to the units of t.

        """
        duration = np.atleast_1d(np.abs(duration))
        if duration.ndim != 1 or duration.size == 0:
            raise ValueError("duration must be 1-dimensional")
        return validate_unit_consistency(self._trel, duration)

    def _validate_period_and_duration(self, period, duration):
        """Private method used to check a set of periods and durations.

        Parameters
        ----------
        period : float, array-like, or `~astropy.units.Quantity` ['time']
            The set of test periods.
        duration : float, array-like, or `~astropy.units.Quantity` ['time']
            The set of durations that will be considered.

        Returns
        -------
        period, duration : array-like or `~astropy.units.Quantity` ['time']
            The inputs reformatted with the correct shapes and units.

        Raises
        ------
        ValueError
            If the units of period or duration cannot be converted to the
            units of t.

        """
        duration = self._validate_duration(duration)
        period = np.atleast_1d(np.abs(period))
        if period.ndim != 1 or period.size == 0:
            raise ValueError("period must be 1-dimensional")
        period = validate_unit_consistency(self._trel, period)

        if not np.min(period) > np.max(duration):
            raise ValueError(
                "The maximum transit duration must be shorter than the minimum period"
            )

        return period, duration

    def _format_results(self, t_ref, objective, period, results):
        """A private method used to wrap and add units to the periodogram.

        Parameters
        ----------
        t_ref : float
            The minimum time in the time series (a reference time).
        objective : str
            The name of the objective used in the optimization.
        period : array-like or `~astropy.units.Quantity` ['time']
            The set of trial periods.
        results : tuple
            The output of one of the periodogram implementations.

        """
        (
            power,
            depth,
            depth_err,
            duration,
            transit_time,
            depth_snr,
            log_likelihood,
        ) = results
        transit_time += t_ref

        if has_units(self._trel):
            transit_time = units.Quantity(transit_time, unit=self._trel.unit)
            transit_time = self._as_absolute_time_if_needed(
                "transit_time", transit_time
            )
            duration = units.Quantity(duration, unit=self._trel.unit)

        if has_units(self.y):
            depth = units.Quantity(depth, unit=self.y.unit)
            depth_err = units.Quantity(depth_err, unit=self.y.unit)

            depth_snr = units.Quantity(depth_snr, unit=units.one)

            if self.dy is None:
                if objective == "likelihood":
                    power = units.Quantity(power, unit=self.y.unit**2)
                else:
                    power = units.Quantity(power, unit=units.one)
                log_likelihood = units.Quantity(log_likelihood, unit=self.y.unit**2)
            else:
                power = units.Quantity(power, unit=units.one)
                log_likelihood = units.Quantity(log_likelihood, unit=units.one)

        return BoxLeastSquaresResults(
            objective,
            period,
            power,
            depth,
            depth_err,
            duration,
            transit_time,
            depth_snr,
            log_likelihood,
        )

    def _t_unit(self):
        if has_units(self._trel):
            return self._trel.unit
        else:
            return 1

    def _y_unit(self):
        if has_units(self.y):
            return self.y.unit
        else:
            return 1


class BoxLeastSquaresResults(dict):
    """The results of a BoxLeastSquares search.

    Attributes
    ----------
    objective : str
        The scalar used to optimize to find the best fit phase, duration, and
        depth. See :func:`BoxLeastSquares.power` for more information.
    period : array-like or `~astropy.units.Quantity` ['time']
        The set of test periods.
    power : array-like or `~astropy.units.Quantity`
        The periodogram evaluated at the periods in ``period``. If
        ``objective`` is:

        * ``'likelihood'``: the values of ``power`` are the
          log likelihood maximized over phase, depth, and duration, or
        * ``'snr'``: the values of ``power`` are the signal-to-noise with
          which the depth is measured maximized over phase, depth, and
          duration.

    depth : array-like or `~astropy.units.Quantity`
        The estimated depth of the maximum power model at each period.
    depth_err : array-like or `~astropy.units.Quantity`
        The 1-sigma uncertainty on ``depth``.
    duration : array-like or `~astropy.units.Quantity` ['time']
        The maximum power duration at each period.
    transit_time : array-like, `~astropy.units.Quantity`, or `~astropy.time.Time`
        The maximum power phase of the transit in units of time. This
        indicates the mid-transit time and it will always be in the range
        (0, period).
    depth_snr : array-like or `~astropy.units.Quantity`
        The signal-to-noise with which the depth is measured at maximum power.
    log_likelihood : array-like or `~astropy.units.Quantity`
        The log likelihood of the maximum power model.

    """

    def __init__(self, *args):
        super().__init__(
            zip(
                (
                    "objective",
                    "period",
                    "power",
                    "depth",
                    "depth_err",
                    "duration",
                    "transit_time",
                    "depth_snr",
                    "log_likelihood",
                ),
                args,
            )
        )

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return "\n".join(
                [k.rjust(m) + ": " + repr(v) for k, v in sorted(self.items())]
            )
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())
