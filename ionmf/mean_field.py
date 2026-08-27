"""Mean-field (neural mass) model, Eq. 25 of the paper.

State: z = [x, V, n, DKi, Kg]
  x   : firing-rate variable, r = R_minus * x / pi  (kHz -> multiply by 1000 for Hz)
  V   : mean membrane potential (mV)
  n, DKi, Kg : as in the single neuron

The V-nullcline of the single neuron is approximated by two parabolas that
meet at V*: for V <= V*  ->  R_minus (V - c_minus)^2 ; for V > V*  ->
R_plus (V - c_plus)^2 (see ``nullcline``). Neurons are all-to-all coupled with
synaptic strength J, reversal potential E, and Lorentzian-distributed
excitability (mean eta, half-width Delta).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import odeint

from . import params as P
from .single_neuron import currents, n_inf


def rhs(z, t, K_bath, J, eta, Delta, coef, par, E=0.0, Input=0.0):
    x, V, n, DKi, Kg = z
    I_Na, I_K, I_Cl, I_pump = currents(V, n, DKi, Kg, par)
    K_o = P.concentrations(DKi, Kg, par)[3]
    Vdot = -(I_Na + I_K + I_Cl + I_pump) / par["Cm"]

    if V <= coef["Vstar"]:
        R, c = coef["R_minus"], coef["c_minus"]
    else:
        R, c = coef["R_plus"], coef["c_plus"]
    dV = Vdot - R * x ** 2 + eta
    dx = Delta + 2.0 * R * (V - c) * x

    r = coef["R_minus"] * x / np.pi + Input     # firing rate (kHz)
    dV += J * r * (E - V)
    dx += -J * r * x
    dn = (n_inf(V, par) - n) / par["tau_n"]
    dDKi = -(par["gamma"] / par["w_i"]) * (I_K - 2.0 * I_pump)
    dKg = par["epsilon"] * (K_bath - K_o)
    return [dx, dV, dn, dDKi, dKg]


def simulate(K_bath, J=1.0, eta=0.0, Delta=1.0, E=0.0,
             z0=(0.1, -15.0, 0.45, -3.5, -12.0), duration_ms=20_000.0,
             dt_ms=1.0, coef=None, par=None, Input=0.0):
    """Integrate the mean-field model with LSODA and return a dict of traces
    (t, x, r [Hz], V, n, DKi, Kg, Kext)."""
    par = P.DEFAULT if par is None else par
    coef = P.PARABOLA if coef is None else coef
    t = np.arange(0.0, duration_ms + dt_ms / 2, dt_ms)
    z = odeint(rhs, list(z0), t,
               args=(float(K_bath), J, eta, Delta, coef, par, E, Input))
    out = {"t": t, "x": z[:, 0], "V": z[:, 1], "n": z[:, 2],
           "DKi": z[:, 3], "Kg": z[:, 4]}
    out["r"] = 1000.0 * coef["R_minus"] * out["x"] / np.pi
    out["Kext"] = P.k_ext(out["DKi"], out["Kg"], par)
    return out
