#!/usr/bin/env python
"""Figure 3 -- figure supplement 1: error of the first-moment closure.

For several [K+]_bath a population of N = 3000 neurons (parameters of Fig. 3)
is simulated and each neuron is classified at every instant from its own
gating variable n_i, using a sliding window of WIN ms:
  quiescent   : max(n_i) over the window < N_LOW
  depolarized : min(n_i) over the window > N_HIGH  (n stays high, no oscillation)
  bursting    : otherwise
The population mean <n> is classified with the same rule; the "error" is the
fraction of neurons whose state differs from the state of the mean.
(a) state fractions, <n> and error versus time for four [K+]_bath
(b) mean error (%) after the transient versus [K+]_bath

Note: the classification code used for the published figure was not archived;
this is a re-implementation of the procedure described in the caption.
"""
import matplotlib.pyplot as plt
import numpy as np

from _common import parser, cached_parallel
from ionmf import population as pop
from ionmf.plotting import save

p = parser(__doc__)
p.add_argument("--N", type=int, default=3000)
p.add_argument("--duration", type=float, default=24_000.0)
p.add_argument("--workers", type=int, default=3, help="parallel population runs")
args = p.parse_args()
N, DUR = (args.N, args.duration) if not args.quick else (300, 6_000.0)
WIN, N_LOW, N_HIGH = 200, 0.15, 0.30     # ms, thresholds on n
KB_A = [5.5, 10.5, 15.5, 27.5]
KB_B = [5.5, 7.5, 11.5, 15.5, 21.5, 24.5, 27.5]


def classify(n):
    """n: (..., time) sampled at 1 ms -> int states (0 quiescent, 1 bursting, 2 depolarized)."""
    from scipy.ndimage import maximum_filter1d, minimum_filter1d
    nmax = maximum_filter1d(n, WIN, axis=-1, mode="nearest")
    nmin = minimum_filter1d(n, WIN, axis=-1, mode="nearest")
    st = np.ones(n.shape, dtype=np.int8)
    st[nmax < N_LOW] = 0
    st[nmin > N_HIGH] = 2
    return st


def run(k):
    s = pop.simulate(N=N, K_bath=k, J=1.0, Delta=1.0, eta_bar=0.0, E=0.0, Vth=-50.0,
                     duration_ms=DUR, dt_ms=0.01, z0=(-15.0, 0.45, -3.5, -12.0), seed=0,
                     record_all=True, report="text")
    st = classify(s["n"])
    st_mean = classify(s["n_mean"][None])[0]
    frac = np.stack([(st == i).mean(0) for i in range(3)])
    err = (st != st_mean[None]).mean(0)
    err[:WIN] = 0.0          # classification undefined during the first window
    return {"t": s["t"], "n_mean": s["n_mean"], "frac": frac, "err": err}


KS = sorted(set(KB_A + KB_B))
res = cached_parallel([(f"figS2_states_N{N}_T{DUR:g}_K{k}", lambda k=k: run(k)) for k in KS],
                      workers=args.workers, use_cache=not args.no_cache)
res = {k: res[f"figS2_states_N{N}_T{DUR:g}_K{k}"] for k in KS}

fig, axes = plt.subplots(2, 2, figsize=(5.6, 3.6))
for ax, k in zip(axes.ravel(), KB_A):
    r = res[k]
    for i, (lab, c) in enumerate((("Quiescent", "tab:blue"), ("Bursting", "tab:orange"),
                                  ("Depolarized", "crimson"))):
        ax.plot(r["t"], r["frac"][i], color=c, lw=0.8, label=lab)
    ax.plot(r["t"], r["err"], "k--", lw=0.8, label="Error")
    ax.set_ylim(0, 1.02); ax.set_xlabel("Time (ms)"); ax.set_ylabel("Fraction of neurons")
    ax.set_title(f"$[K^+]_{{bath}}$ = {k}")
    ax2 = ax.twinx()
    ax2.plot(r["t"], r["n_mean"], color="tab:green", lw=0.5)
    ax2.set_ylabel(r"$\langle n \rangle$", color="tab:green"); ax2.tick_params(axis="y", colors="tab:green")
axes[0, 0].legend(frameon=False, loc="center right")
fig.tight_layout()
save(fig, args.out, "FigS2a_state_fractions")

fig, ax = plt.subplots(figsize=(2.2, 1.9))
cut = int(min(5000, DUR / 4))
errs = [100 * res[k]["err"][cut:].mean() for k in KB_B]
ax.plot(KB_B, errs, "k.-", lw=1)
ax.set_xlabel("$[K^+]_{bath}$"); ax.set_ylabel("Deviating Neurons (%)"); ax.set_ylim(0, None)
save(fig, args.out, "FigS2b_error_vs_Kbath")
