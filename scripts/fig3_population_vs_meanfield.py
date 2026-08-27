#!/usr/bin/env python
"""Figure 3 -- mean-field model versus a network of N = 3000 HH-type neurons.

(a) raster plot of the population at [K+]_bath = 15.5 and the instantaneous
    distribution of membrane potentials during a seizure-like event
(b) for [K+]_bath in {7.5, 11.5, 15.5, 21.5, 27.5}: population mean V and
    [K+]_ext (N = 3000, J = 1, Delta = 1, eta = 0) versus the mean-field model
    run at the effective [K+]_bath^eff of panel (c)
(c) the effective [K+]_bath^eff used for the mean field, for J = 1 and J = 0.1
    (values obtained in the paper by visual inspection of the match; they are
    stored here as data, see KEFF below)

Population runs take ~30-60 min each at N = 3000 (dt = 0.01 ms, 24 s);
they are cached in cache/.  Use --quick for a fast, non-quantitative check.
"""
import matplotlib.pyplot as plt
import numpy as np

from _common import parser, cached, cached_parallel
from ionmf import population as pop, mean_field as mf, params as P
from ionmf.plotting import strip, save, POP_COLOR, MF_COLOR

p = parser(__doc__)
p.add_argument("--N", type=int, default=3000)
p.add_argument("--duration", type=float, default=24_000.0, help="ms")
p.add_argument("--workers", type=int, default=3, help="parallel population runs")
args = p.parse_args()
N, DUR = (args.N, args.duration) if not args.quick else (300, 6_000.0)
Z0 = (-15.0, 0.45, -3.5, -12.0)

# [K+]_bath -> [K+]_bath^eff (Fig. 3c), from the original analysis notebook
KEFF = {
    1.0: dict(zip([3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5, 17.5, 19.5, 21.5, 23.5, 25.5, 27.5],
                  [3.5, 5.5, 8.9, 9.6, 11.1, 13.5, 15.1, 17.5, 19.5, 19.7, 19.7, 22.5, 23.8])),
    0.1: dict(zip([3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5, 17.5, 19.5, 21.5, 23.5, 25.5, 27.5],
                  [3.5, 5.5, 7.4, 9.6, 11.1, 13.5, 15.3, 17.5, 19.7, 20.1, 20.08, 24.0, 25.8])),
}
KB = [7.5, 11.5, 15.5, 21.5, 27.5]
J = 1.0


def run_pop(k, record_all=False):
    return pop.simulate(N=N, K_bath=k, J=J, Delta=1.0, eta_bar=0.0, E=0.0, Vth=-50.0,
                        duration_ms=DUR, dt_ms=0.01, z0=Z0, seed=0,
                        record_all=record_all, report="text")


# run all population simulations first (in parallel, cached)
jobs = [(f"fig3_pop_N{N}_T{DUR:g}_K15.5_full", lambda: run_pop(15.5, record_all=True))]
jobs += [(f"fig3_pop_N{N}_T{DUR:g}_K{k}", lambda k=k: run_pop(k)) for k in KB]
pops = cached_parallel(jobs, workers=args.workers, use_cache=not args.no_cache)

# ---------------------------------------------------------------- panel a
sa = pops[f"fig3_pop_N{N}_T{DUR:g}_K15.5_full"]
V = sa["V"]                                     # (N, time) sampled at 1 ms
nwin = int(min(9000, V.shape[1] * 0.75))         # last 9 s, as in the original figure
V = V[:, -nwin:]
t = sa["t"][-nwin:] / 1000
spikes = V > 0
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.0, 1.6), gridspec_kw=dict(width_ratios=[2, 1]))
ax1.imshow(spikes, aspect="auto", interpolation="none", cmap="gray_r",
           extent=[t[0], t[-1], 0, V.shape[0]], origin="lower")
ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Neuron ID"); ax1.set_title("Neurons rasterplot")
# instants for the histograms: inside a seizure-like event (most spiking
# neurons) and the last quiet sample (no spikes) before it
nsp = spikes.sum(0)
tsz = int(np.argmax(nsp))
tq = int(np.argmin(V.mean(0)))        # inter-event hyperpolarized instant
ax1.axvline(t[tq], color="tab:blue", ls="--", lw=0.6)
ax1.axvline(t[tsz], color="tomato", ls="--", lw=0.6)
ax2.hist(V[:, tq], 40, color="tab:blue", alpha=0.9)
ax2.hist(V[:, tsz], 40, color="tomato", alpha=0.9)
ax2.set_xlabel("V"); ax2.set_yticks([]); ax2.set_title("Instantaneous membrane\npotential distribution")
strip(ax2, bottom=False, left=True)
save(fig, args.out, "Fig3a_raster_histogram")

# ---------------------------------------------------------------- panel b
fig, axes = plt.subplots(3, len(KB), figsize=(6.4, 2.6), sharex="col")
for j, k in enumerate(KB):
    sp = pops[f"fig3_pop_N{N}_T{DUR:g}_K{k}"]
    keff = KEFF[J].get(k, k)
    sm = cached(f"fig3_mf_J{J}_T{DUR:g}_Keff{keff}", lambda keff=keff: mf.simulate(
        keff, J=J, eta=0.0, Delta=1.0, E=0.0, z0=(0.1,) + Z0, duration_ms=DUR, dt_ms=1.0),
        not args.no_cache)
    ncut = int(min(4000, DUR / 6))          # drop the initial transient
    tp, tm = sp["t"][ncut:] / 1000, sm["t"][ncut:] / 1000
    axes[0, j].plot(tp, sp["V_mean"][ncut:], color=POP_COLOR, lw=0.5)
    axes[1, j].plot(tm, sm["V"][ncut:], color=MF_COLOR, lw=0.5)
    axes[2, j].plot(tp, sp["Kext_mean"][ncut:], color=POP_COLOR, lw=0.7, label="population")
    axes[2, j].plot(tm, sm["Kext"][ncut:], color=MF_COLOR, lw=0.7, label="mean field")
    axes[0, j].set_title(f"$[K^+]_{{bath}}$ = {k}\n$[K^+]_{{bath}}^{{eff}}$ = {keff}", fontsize=7)
    for ax in axes[:, j]:
        strip(ax, bottom=False)
    axes[2, j].set_xlabel("time (s)")
axes[0, 0].set_ylabel("$V_{pop}$"); axes[1, 0].set_ylabel("$V_{MF}$"); axes[2, 0].set_ylabel("$[K^+]_{ext}$")
axes[2, 0].legend(frameon=False, loc="lower right")
save(fig, args.out, "Fig3b_population_vs_meanfield")

# ---------------------------------------------------------------- panel c
fig, ax = plt.subplots(figsize=(1.8, 1.8))
for jj_, mk in ((1.0, ".-"), (0.1, ".-")):
    ks = sorted(KEFF[jj_])
    ax.plot(ks, [KEFF[jj_][k] for k in ks], mk, lw=1, ms=4, label=f"J={jj_:g}")
ax.plot([3.5, 27.5], [3.5, 27.5], "--", color="slategray", lw=0.8)
ax.set_xlabel("$[K^+]_{bath}$"); ax.set_ylabel("$[K^+]_{bath}^{eff}$"); ax.set_title("Reparametrization")
ax.legend(frameon=False)
save(fig, args.out, "Fig3c_reparametrization")
