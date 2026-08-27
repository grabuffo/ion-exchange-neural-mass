#!/usr/bin/env python
"""Figure 6 -- six structurally connected neural masses.

(a) connectivity weights (data/connectivity/6x6full.zip) and graph
(b) firing rates with global coupling G = 0: only node D ([K+]_bath = 15.5)
    shows seizure-like bursts; the others ([K+]_bath = 5.5) are silent
(c) G = 100: the pathological activity of D propagates to the network
Mean-field parameters: J = 0.08, Delta = 1, eta = 0, R- = 0.02, R+ = -0.1,
c- = -44.24, c+ = -20.4, V* = -31 (Supplementary File 1, "Fig. 6b-c" column).
"""
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from _common import parser, cached, DATA_DIR
from ionmf import network as nw, params as P
from ionmf.plotting import strip, save

args = parser(__doc__).parse_args()
W, L, _ = nw.load_connectivity(os.path.join(DATA_DIR, "connectivity", "6x6full.zip"))
labels = list("ABCDEF")
coef = P.parabola(R_minus=0.02, R_plus=-0.1, c_minus=-44.24, c_plus=-20.4, Vstar=-31.0)
KB = np.array([5.5, 5.5, 5.5, 15.5, 5.5, 5.5])
DUR = 20_000.0 if not args.quick else 5_000.0
cmap = plt.get_cmap("viridis", 10)
cols = [cmap(3), cmap(2), cmap(7), "tomato", cmap(4), cmap(1)]

# ---------------------------------------------------------------- panel a
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.2, 1.9), gridspec_kw=dict(width_ratios=[1.1, 1]))
im = ax1.imshow(W, cmap="viridis")
ax1.set_xticks(range(6)); ax1.set_yticks(range(6))
ax1.set_xticklabels(labels); ax1.set_yticklabels(labels); ax1.set_title("Weights")
fig.colorbar(im, ax=ax1, shrink=0.8)
G = nx.from_numpy_array(W)
pos = nx.circular_layout(G)
widths = 25 * np.array([W[i, j] for i, j in G.edges()])
nx.draw(G, pos, ax=ax2, labels=dict(enumerate(labels)), width=widths, node_size=260,
        node_color=["#2a9d8f"] * 3 + ["tomato"] + ["#2a9d8f"] * 2, font_color="w",
        font_size=7, font_weight="bold", edge_color="#4a4a4a")
save(fig, args.out, "Fig6a_connectivity")

# ---------------------------------------------------------------- panels b-c
for G_, name in ((0.0, "b"), (100.0, "c")):
    sim = cached(f"fig6_network_G{G_:g}", lambda G_=G_: nw.simulate(
        W, G_, KB, J=0.08, eta=0.0, Delta=1.0, E=0.0, coef=coef,
        z0=(0.1, -15.0, 0.45, -5.0, -16.0), duration_ms=DUR, dt_ms=0.1, record_every=10),
        not args.no_cache)
    fig, axes = plt.subplots(6, 1, figsize=(3.3, 2.4), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(sim["t"] / 1000, sim["r"][:, i], color=cols[i], lw=0.7)
        ax.set_ylabel(labels[i] + " ", rotation=0, labelpad=4, va="center")
        strip(ax, bottom=(i < 5))
    axes[0].set_title(f"Firing rate, $G = {G_:g}$")
    axes[-1].set_xlabel("time (s)")
    save(fig, args.out, f"Fig6{name}_rates_G{G_:g}")
