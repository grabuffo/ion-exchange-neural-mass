#!/usr/bin/env python
"""Figure 4 -- slow extracellular-potassium oscillations with bursts on top.

(a) mean-field model with slow ion dynamics: C_m = 16, tau_n = 8,
    epsilon = 1e-4, gamma = 2.5e-4, J = 0.01, R- = 0.001, R+ = -0.01
    ([K+]_bath = 15.5, 15 min of simulated time; the last 12 min are shown)
(b) network of N = 3000 HH-type neurons with rescaled parameters
    (C_m = 10, tau_n = 2.4, J = 1, Delta = 1, [K+]_bath = 15.5) -- shorter
    bursting period, same qualitative picture
(c, d) in vitro LFP + [K+]_ext recordings: plotted if csv files are found in
    data/in_vitro/ (see data/in_vitro/README.md), otherwise skipped
(e) an emergent mean-field regime with isolated bursts in the up state
    (J = 2, R- = 0.5, R+ = -0.5, [K+]_bath = 15.5; Supplementary File 1,
    column labelled "Fig. 5c" in the preprint numbering)
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from _common import parser, cached, DATA_DIR
from ionmf import population as pop, mean_field as mf, params as P
from ionmf.plotting import strip, save, scalebar, V_COLOR, K_COLOR

p = parser(__doc__)
p.add_argument("--N", type=int, default=3000)
args = p.parse_args()
N = args.N if not args.quick else 300


def two_traces(t, V, K, title, bar_s, name):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.2, 1.9), sharex=True)
    a1.plot(t, V, color=V_COLOR, lw=0.5); a1.set_ylabel("V"); a1.set_title(title)
    a2.plot(t, K, color=K_COLOR, lw=0.8); a2.set_ylabel("$[K^+]_{ext}$")
    strip(a1); strip(a2)
    scalebar(a2, t[0], bar_s, f"{bar_s:g} s" if bar_s < 60 else f"{bar_s/60:g} min")
    save(fig, args.out, name)


# ---------------------------------------------------------------- panel a
par_a = P.make(Cm=16.0, tau_n=8.0, gamma=0.00025, epsilon=0.0001)
coef_a = P.parabola(R_minus=0.001, R_plus=-0.01)
dur_a = 15 * 60_000.0 if not args.quick else 3 * 60_000.0
sa = cached(f"fig4a_mf_slow_{dur_a:g}", lambda: mf.simulate(
    15.5, J=0.01, eta=0.0, Delta=1.0, E=0.0, z0=(36.8, -70.8, 0.053, -0.231, 7.93),
    duration_ms=dur_a, dt_ms=1.0, coef=coef_a, par=par_a), not args.no_cache)
n_show = int(len(sa["t"]) * 12 / 15)
two_traces(sa["t"][-n_show:] / 1000, sa["V"][-n_show:], sa["Kext"][-n_show:],
           "Mean-field simulation", 120, "Fig4a_meanfield_slow")

# ---------------------------------------------------------------- panel b
par_b = P.make(Cm=10.0, tau_n=2.4)
dur_b = 20_000.0 if not args.quick else 4_000.0
sb = cached(f"fig4b_pop_N{N}_T{dur_b:g}", lambda: pop.simulate(
    N=N, K_bath=15.5, J=1.0, Delta=1.0, eta_bar=0.0, E=0.0, Vth=-50.0,
    duration_ms=dur_b, dt_ms=0.01, z0=(-81.78, 0.03, -4.92, -14.88), par=par_b,
    seed=0, report="text"), not args.no_cache)
cut = int(len(sb["t"]) * 0.2)
two_traces(sb["t"][cut:] / 1000, sb["V_mean"][cut:], sb["Kext_mean"][cut:],
           "Neural network simulation", 1, "Fig4b_network_rescaled")

# ---------------------------------------------------------------- panels c, d
files = sorted(glob.glob(os.path.join(DATA_DIR, "in_vitro", "*.csv")))
if not files:
    print("panels c-d: no in vitro csv found in data/in_vitro/ -- skipped")
for f in files:
    d = np.genfromtxt(f, delimiter=",", names=True)
    name = os.path.splitext(os.path.basename(f))[0]
    two_traces(d["time_s"], d["lfp"], d["k_ext_mM"], f"In-vitro experiment ({name})",
               60 if d["time_s"][-1] > 300 else 5, f"Fig4cd_invitro_{name}")

# ---------------------------------------------------------------- panel e
coef_e = P.parabola(R_minus=0.5, R_plus=-0.5)
se = cached("fig4e_mf_emergent_J2", lambda: mf.simulate(
    15.5, J=2.0, eta=0.0, Delta=1.0, E=0.0, z0=(0.1, -15.0, 0.45, -5.0, -16.0),
    duration_ms=20_000.0, dt_ms=1.0, coef=coef_e), not args.no_cache)
sl = slice(-6000, None)
two_traces(se["t"][sl] / 1000, se["V"][sl], se["Kext"][sl], "Mean-field simulation", 6,
           "Fig4e_meanfield_emergent")
