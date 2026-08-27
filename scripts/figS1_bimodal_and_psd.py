#!/usr/bin/env python
"""Figure 2 -- figure supplement 1: nullcline geometry and bimodality.

(a) a network in which a sub-population switches from sub- to supra-threshold
    activity shows a bimodal membrane-potential distribution. This is the
    N = 3000, [K+]_bath = 15.5, J = 1, Delta = 1 population of Fig. 3 (same
    cached simulation as fig3_population_vs_meanfield.py)
(b) power spectral density of the mean-field V while varying one parabola
    coefficient at a time (c-, R-, c+, R+; [K+]_bath = 15.5, J = 1, Delta = 1)
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from _common import parser, cached
from ionmf import population as pop, mean_field as mf, params as P
from ionmf.plotting import strip, save

args = parser(__doc__).parse_args()

# ---------------------------------------------------------------- panel a
N, dur = (3000, 24_000.0) if not args.quick else (300, 6_000.0)
sa = cached(f"fig3_pop_N{N}_T{dur:g}_K15.5_full", lambda: pop.simulate(
    N=N, K_bath=15.5, J=1.0, Delta=1.0, eta_bar=0.0, E=0.0, Vth=-50.0,
    duration_ms=dur, dt_ms=0.01, z0=(-15.0, 0.45, -3.5, -12.0), record_all=True,
    seed=0, report="text"), not args.no_cache)
V, t = sa["V"], sa["t"] / 1000
nwin = int(min(3000, V.shape[1] * 0.5))          # 3 s window
V, t = V[:, -nwin:], t[-nwin:]
spk = V > 0
# instant where the population is split between sub- and supra-threshold
# neurons = largest across-neuron variance of V (bimodal distribution)
t_bi = int(np.argmax(V.var(0)))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(2.4, 3.6), gridspec_kw=dict(hspace=0.9))
ax1.imshow(spk, aspect="auto", interpolation="none", cmap="gray_r",
           extent=[t[0], t[-1], 0, V.shape[0]], origin="lower")
ax1.axvline(t[t_bi], color="tomato", ls="--", lw=0.8)
ax1.set_ylabel("Neuron ID"); ax1.set_xlabel("Time (s)"); ax1.set_title("Neurons rasterplot")
ax2.hist(V[:, t_bi], 60, color="tomato")
ax2.set_title("Instantaneous membrane\npotential distribution"); ax2.set_xlabel("V")
strip(ax2, bottom=False, left=True)
save(fig, args.out, "FigS1a_bimodal_distribution")

# ---------------------------------------------------------------- panel b
n = 31 if not args.quick else 7
SWEEPS = {"c_minus": np.linspace(-70, -40, n), "R_minus": np.logspace(-3, 0, n),
          "c_plus": np.linspace(-27, -10, n), "R_plus": -np.logspace(-3, 0, n)[::-1]}
TEX = {"c_minus": "$c_-$", "R_minus": "$R_-$", "c_plus": "$c_+$", "R_plus": "$R_+$"}
dur_b = 20_000.0 if not args.quick else 6_000.0
fig, axes = plt.subplots(2, 2, figsize=(4.6, 3.2))
for ax, (key, vals) in zip(axes.T.ravel(), SWEEPS.items()):
    spec = []
    for v in vals:
        s = cached(f"figS1b_mf_T{dur_b:g}_{key}_{v:.4g}", lambda v=v: mf.simulate(
            15.5, J=1.0, eta=0.0, Delta=1.0, E=0.0, z0=(0.1, -15.0, 0.45, -3.5, -12.0),
            duration_ms=dur_b, dt_ms=1.0, coef=P.parabola(**{key: float(v)})), not args.no_cache)
        f, pxx = welch(s["V"], fs=1000, nperseg=1024)
        spec.append(pxx)
    spec = np.log(np.array(spec)).T
    im = ax.imshow(spec, aspect="auto", cmap="plasma", interpolation="none", origin="lower",
                   extent=[0, len(vals), f[0], f[-1]])
    step = max(1, len(vals) // 6)
    ax.set_xticks(np.arange(len(vals))[::step] + 0.5)
    ax.set_xticklabels([f"{x:.3g}" for x in vals[::step]], rotation=45)
    ax.set_xlabel(TEX[key]); ax.set_ylabel("freqs (Hz)")
    fig.colorbar(im, ax=ax, label="log(PSD)")
fig.tight_layout()
save(fig, args.out, "FigS1b_psd_parabola_sweeps")
