"""Synthetic wide-exposure transit test for the exptime-aware BLS.

A short transit (duration 0.12 d) is observed with exposures of 0.5 d,
so the measured dips are smeared to 0.12/0.5 = 24% of the physical depth.
The classical (instantaneous) BLS recovers only the smeared depth, while
the exposure-aware mode recovers the injected physical depth.
"""

import numpy as np

from astropy.timeseries.periodograms.bls import BoxLeastSquares
from astropy.timeseries.periodograms.bls.methods import window_overlap_fraction

period_true = 3.5
duration_true = 0.12
depth_true = 0.02
t0_true = 1.1
exptime = 0.5  # exposure time >> transit duration

t = np.arange(exptime / 2, 70.0, exptime)
dy = 0.001 * np.ones_like(t)
et = np.full_like(t, exptime)

# 1. noiseless injection: the exposure-aware depth estimate is exact
y_noiseless = 1.0 - depth_true * window_overlap_fraction(
    t, et, period_true, duration_true, t0_true
)
bls = BoxLeastSquares(t, y_noiseless, dy, exptime=exptime)
stats0 = bls.compute_stats(period_true, duration_true, t0_true)
print(
    "noiseless:  depth=%.6f (injected %.3f, smeared %.5f)"
    % (stats0['depth'][0], depth_true, depth_true * duration_true / exptime)
)
assert abs(stats0['depth'][0] - depth_true) < 1e-10
model0 = bls.model(t, period_true, duration_true, t0_true, exptime=exptime)
assert np.max(np.abs(model0 - y_noiseless)) < 1e-10

# 2. noisy light curve
rng = np.random.default_rng(7)
y = y_noiseless + dy * rng.standard_normal(len(t))

# classical BLS: instant sampling assumed -> biased (smeared) depth
bls_classical = BoxLeastSquares(t, y, dy)
res_c = bls_classical.power([period_true], [duration_true], oversample=50)
k = np.argmax(res_c.power)
print(
    "classical:  depth=%.5f +- %.5f (biased by smearing)"
    % (res_c.depth[k], res_c.depth_err[k])
)

# exposure-aware BLS: recovers the physical depth
bls_exp = BoxLeastSquares(t, y, dy, exptime=exptime)
res_e = bls_exp.power([period_true], [duration_true], oversample=50)
j = np.argmax(res_e.power)
print(
    "exptime:    depth=%.5f +- %.5f at t0=%.3f (SNR %.1f)"
    % (res_e.depth[j], res_e.depth_err[j], res_e.transit_time[j], res_e.depth_snr[j])
)
assert abs(res_e.depth[j] - depth_true) < 3 * res_e.depth_err[j]
assert abs(res_e.depth[j] - depth_true) < abs(res_c.depth[k] - depth_true)

# period search recovers the injected period
periods = bls_exp.autoperiod(
    [0.1, 0.12, 0.16], minimum_period=2.0, maximum_period=6.0, frequency_factor=5.0
)
res_p = bls_exp.power(periods, [0.1, 0.12, 0.16])
i = np.argmax(res_p.power)
print(
    "period search: P=%.4f (injected %.2f), depth=%.5f"
    % (res_p.period[i], period_true, res_p.depth[i])
)
assert abs(res_p.period[i] - period_true) < 0.02

# 3. compute_stats, model and transit_mask use the same window semantics
stats = bls_exp.compute_stats(period_true, duration_true, res_e.transit_time[j])
print("stats:      depth=%.5f +- %.5f" % stats["depth"])
assert abs(stats["depth"][0] - depth_true) < 3 * stats["depth"][1]

model = bls_exp.model(
    t, period_true, duration_true, res_e.transit_time[j], exptime=exptime
)
print("model rms:  %.5f (noise level %.3f)" % (np.std(y - model), dy[0]))
assert np.std(y - model) < 1.5 * dy[0]

mask = bls_exp.transit_mask(
    t, period_true, duration_true, res_e.transit_time[j], exptime=exptime
)
print("in-transit: %d of %d exposures" % (mask.sum(), len(t)))

# 4. zero exposure times degrade to the classical results
bls_zero = BoxLeastSquares(t, y, dy, exptime=np.zeros_like(t))
res_z = bls_zero.power([period_true], [duration_true], oversample=50)
np.testing.assert_allclose(res_z.power, res_c.power, rtol=1e-8, equal_nan=False)
np.testing.assert_allclose(res_z.depth, res_c.depth, rtol=1e-8, equal_nan=False)
np.testing.assert_allclose(
    bls_zero.model(t, period_true, duration_true, res_e.transit_time[j]),
    bls_classical.model(t, period_true, duration_true, res_e.transit_time[j]),
    rtol=1e-8,
)
print("zero-exptime equivalence: OK")

print("ALL CHECKS PASSED")
