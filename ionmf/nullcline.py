"""V-nullcline geometry and the piecewise-quadratic fit (Fig. 2c-d).

For each [K+]_bath the single neuron is simulated; the slow variables are
frozen (n at its maximum, DKi and Kg at their means over the run) and dV/dt is evaluated on a grid of V. The resulting cubic-like curve
is approximated by two parabolas: one centred on the minimum (c_minus,
curvature R_minus) and one on the maximum (c_plus, R_plus), meeting with a
continuous derivative at the inflection point V*. Averaging the coefficients
over [K+]_bath gives the values in ``params.PARABOLA``.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import argrelextrema

from . import params as P
from .single_neuron import simulate, v_dot


def vdot_curve(sim, par, V_grid=None):
    """dV/dt on ``V_grid`` with the slow variables frozen from ``sim``:
    n at its maximum, DKi and Kg at their mean over the whole simulation
    (this is what the original analysis did and what yields the published
    coefficients; using only the late part of the trace changes them by
    ~10%)."""
    V_grid = np.arange(-80, 0) if V_grid is None else np.asarray(V_grid, float)
    n = np.max(sim["n"])
    DKi = np.mean(sim["DKi"])
    Kg = np.mean(sim["Kg"])
    return V_grid, v_dot(V_grid, n, DKi, Kg, par)


def cubic_fit(vs, cubic):
    """Fit two parabolas to a cubic-shaped dV/dt(V) curve.

    Returns (Vstar, c_minus, I_minus, R_minus, c_plus, I_plus, R_plus).
    """
    cmid = int(argrelextrema(cubic, np.less)[0][0])
    cpid = int(argrelextrema(cubic, np.greater)[0][0])
    cm, cp = vs[cmid], vs[cpid]
    im, ip = cubic[cmid], cubic[cpid]
    curvature = np.gradient(np.gradient(cubic[30:-5]))
    signchange = (np.diff(np.sign(curvature)) != 0) * 1
    vst = vs[30 + np.where(signchange == 1)[0]][0]
    pm = ip - im
    # the two parabolas meet at V* with continuous derivative
    rp = pm / ((cp - cm) * (vst - cp))
    rm = rp * (vst - cp) / (vst - cm)
    return vst, cm, im, rm, cp, ip, rp


def fit_parabolas(K_bath_vec=None, par=None, duration_ms=20_000.0, dt_ms=0.1,
                  z0=(-15.0, 0.45, -3.5, -12.0), verbose=False):
    """Run the single neuron for each K_bath, fit parabolas, return
    (per-K_bath table, averaged coefficients dict, list of (V, Vdot) curves)."""
    par = P.DEFAULT if par is None else par
    K_bath_vec = np.linspace(3.5, 27.5, 25) if K_bath_vec is None else K_bath_vec
    rows, curves = [], []
    for k in K_bath_vec:
        sim = simulate(k, z0=z0, duration_ms=duration_ms, dt_ms=dt_ms, par=par)
        V, Vd = vdot_curve(sim, par)
        curves.append((V, Vd))
        rows.append((k,) + cubic_fit(V, Vd))
        if verbose:
            print("K_bath=%5.1f  V*=%6.1f c-=%6.1f R-=%6.3f c+=%6.1f R+=%6.3f" %
                  (k, rows[-1][1], rows[-1][2], rows[-1][4], rows[-1][5], rows[-1][7]))
    tab = np.array(rows)
    coef = {"Vstar": float(np.mean(tab[:, 1])), "c_minus": float(np.mean(tab[:, 2])),
            "R_minus": float(np.mean(tab[:, 4])), "c_plus": float(np.mean(tab[:, 5])),
            "R_plus": float(np.mean(tab[:, 7]))}
    return tab, coef, curves
