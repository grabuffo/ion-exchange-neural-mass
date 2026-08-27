#!/usr/bin/env python
"""Figure 2 -- single HH-type neuron driven by [K+]_bath.

(a) V and [K+]_ext time series for five [K+]_bath values (bursting, tonic
    spiking, seizure-like events, sustained ictal activity, depolarization block)
(b) 3-D trajectory (V, n, [K+]_ext) for [K+]_bath = 15.5
(c) dV/dt versus V for [K+]_bath in [3.5, 27.5] with slow variables frozen
(d) example of the piecewise-quadratic (two parabolas) fit of one such curve
The averaged parabola coefficients (c-, c+, R-, R+, V*) are printed and saved.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from _common import parser, cached, ROOT
from ionmf import single_neuron as sn, nullcline as nc, params as P
from ionmf.plotting import strip, save, V_COLOR, K_COLOR

args = parser(__doc__).parse_args()
par = P.DEFAULT
Z0 = (-15.0, 0.45, -3.5, -12.0)      # initial condition used for the figure
# At [K+]_bath = 24.5 the neuron is bistable: from Z0 it spikes for ~2 s and
# then settles in depolarization block, whereas if [K+]_ext already equals the
# bath value (Kg = 9.2) it stays in the sustained-spiking ("ictal") state
# shown in the paper.  The latter initial condition is used for that panel.
Z0_K = {24.5: (-15.0, 0.45, -3.5, 9.2)}
DT = 0.1                             # ms
DUR = 20_000.0 if not args.quick else 8_000.0

# ---------------------------------------------------------------- panel a
KB = [7.5, 10.5, 15.5, 24.5, 27.5]
LABELS = ["Bursting", "Tonic Spiking", "Seizure-like events",
          "Sustained ictal activity", "Depolarization block"]
# time windows shown (ms), as in the original figure: the last 10 s for 7.5,
# the last 1 s for 10.5, 10 s before the end for 15.5, the first 10 s
# (including the transient to the depolarized state) for 24.5 and 27.5
WINDOW = {7.5: (-10_000, None), 10.5: (-1_000, None), 15.5: (-10_500, -1_500),
          24.5: (0, 10_000), 27.5: (0, 10_000)}
sims = {}
for k in KB:
    z0 = Z0_K.get(k, Z0)
    sims[k] = cached(f"fig2_single_K{k}_Kg{z0[3]:g}", lambda k=k, z0=z0: sn.simulate(
        k, z0=z0, duration_ms=DUR, dt_ms=DT), not args.no_cache)

fig = plt.figure(figsize=(2.8, 7.0))
gs = fig.add_gridspec(len(KB), 1, hspace=0.55)
for i, (k, lab) in enumerate(zip(KB, LABELS)):
    s = sims[k]
    a, b = WINDOW[k]
    scale = DUR / 20_000.0                       # --quick shortens the run
    a = None if a is None else int(a * scale / DT)
    b = None if b is None else int(b * scale / DT)
    sl = slice(a, b)
    t = (s["t"][sl] - s["t"][sl][0]) / 1000
    sub = gs[i].subgridspec(2, 1, hspace=0.15)
    ax = fig.add_subplot(sub[0])
    ax.plot(t, s["V"][sl], color=V_COLOR, lw=0.5)
    ax.set_title(f"$[K^+]_{{bath}}$ = {k}: {lab}", fontsize=7, pad=2)
    ax.set_ylabel("V", labelpad=1)
    strip(ax)
    ax = fig.add_subplot(sub[1])
    ax.plot(t, s["Kext"][sl], color=K_COLOR, lw=0.8)
    ax.set_ylabel("$[K^+]_{ext}$", labelpad=1)
    strip(ax, bottom=False)
    ax.set_xlabel("time (s)", labelpad=1)
save(fig, args.out, "Fig2a_single_neuron_regimes")

# ---------------------------------------------------------------- panel b
s = sims[15.5]
i0, i1 = int(0.475 * len(s["V"])), int(0.6 * len(s["V"]))   # ~2.5 s window as in the paper
fig = plt.figure(figsize=(3.2, 2.2))
ax = fig.add_subplot(111, projection="3d")
ax.plot3D(s["V"][i0:i1], s["n"][i0:i1], s["Kext"][i0:i1], color="darkslategray", lw=0.35)
ax.view_init(22, -198)
ax.set_box_aspect((1, 1, 0.25))
for a in (ax.xaxis, ax.yaxis, ax.zaxis):
    a.set_pane_color((1, 1, 1, 0))
    a._axinfo["grid"]["color"] = (1, 1, 1, 0)
ax.set_xlabel("V"); ax.set_ylabel("n"); ax.set_zlabel("$[K^+]_{ext}$")
save(fig, args.out, "Fig2b_trajectory_3d")

# ---------------------------------------------------------------- panels c-d
KBV = np.linspace(3.5, 27.5, 25) if not args.quick else np.linspace(3.5, 27.5, 7)
tab, coef, curves = nc.fit_parabolas(KBV, par=par, duration_ms=DUR, dt_ms=DT, z0=Z0,
                                     verbose=True)
print("averaged parabola coefficients:", json.dumps(coef, indent=1))
with open(os.path.join(args.out, "Fig2_parabola_coefficients.json"), "w") as f:
    json.dump({"K_bath": list(map(float, KBV)), "mean": coef,
               "columns": ["K_bath", "Vstar", "c_minus", "I_minus", "R_minus",
                           "c_plus", "I_plus", "R_plus"],
               "table": tab.tolist()}, f, indent=1)

fig, ax = plt.subplots(figsize=(2.6, 1.7))
for V, Vd in curves:
    ax.plot(V, Vd, color="darkslategray", lw=0.6)
ax.set_xlabel("V (mV)"); ax.set_ylabel("dV/dt")
ax.annotate("$[K^+]_{bath}$", xy=(-45, curves[-1][1][35]), xytext=(-20, curves[-1][1][35] * 1.4),
            arrowprops=dict(arrowstyle="->", lw=0.6), fontsize=6)
strip(ax, bottom=False)
save(fig, args.out, "Fig2c_vdot_vs_Kbath")

# panel d: one curve (K_bath=15.5) with its two fitted parabolas
i = int(np.argmin(np.abs(KBV - 15.5)))
V, Vd = curves[i]
vst, cm, im, rm, cp, ip, rp = tab[i, 1:]
fig, ax = plt.subplots(figsize=(2.6, 1.7))
ax.plot(V, Vd, color="darkslategray", lw=1)
Vl = np.linspace(-80, vst, 100); Vr = np.linspace(vst, 0, 100)
ax.plot(Vl, im + rm * (Vl - cm) ** 2, "--", color="tab:blue", lw=0.8, label="$R_-(V-c_-)^2$")
ax.plot(Vr, ip + rp * (Vr - cp) ** 2, "--", color="tab:red", lw=0.8, label="$R_+(V-c_+)^2$")
for xx, name in ((cm, "$c_-$"), (cp, "$c_+$"), (vst, "$V^*$")):
    ax.axvline(xx, color="k", lw=0.4, ls=":")
    ax.text(xx, ax.get_ylim()[0], name, ha="center", va="top", fontsize=6)
ax.set_ylim(min(Vd) * 1.5, max(Vd) * 1.5)
ax.legend(frameon=False, loc="upper left")
ax.set_xlabel("V (mV)"); ax.set_ylabel("dV/dt")
strip(ax, bottom=False)
save(fig, args.out, "Fig2d_parabola_fit")
