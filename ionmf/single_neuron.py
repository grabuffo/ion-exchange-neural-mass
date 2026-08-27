"""Single Hodgkin-Huxley-type neuron driven by ion exchange (Eq. 1).

State: z = [V, n, DKi, Kg]
  V   : membrane potential (mV)
  n   : K+ gating variable
  DKi : variation of intracellular K+ (mM)
  Kg  : K+ buffered by glia / exchanged with the bath (mM)
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import odeint

from . import params as P


def m_inf(V, par):
    return 1.0 / (1.0 + np.exp((par["Cmna"] - V) / par["DCmna"]))


def n_inf(V, par):
    return 1.0 / (1.0 + np.exp((par["Cnk"] - V) / par["DCnk"]))


def h(n, par):
    return 1.1 - 1.0 / (1.0 + np.exp(par["DChn"] * (n - par["Chn"])))


def currents(V, n, DKi, Kg, par):
    """Return (I_Na, I_K, I_Cl, I_pump)."""
    K_i, Na_i, Na_o, K_o = P.concentrations(DKi, Kg, par)
    I_K = (par["g_Kl"] + par["g_K"] * n) * (V - P.THERMAL_V * np.log(K_o / K_i))
    I_Na = (par["g_Nal"] + par["g_Na"] * m_inf(V, par) * h(n, par)) * (
        V - P.THERMAL_V * np.log(Na_o / Na_i))
    I_Cl = par["g_Cl"] * (V + P.THERMAL_V * np.log(par["Cl_o0"] / par["Cl_i0"]))
    I_pump = par["rho"] / ((1.0 + np.exp((par["Cnap"] - Na_i) / par["DCnap"]))
                           * (1.0 + np.exp((par["Ckp"] - K_o) / par["DCkp"])))
    return I_Na, I_K, I_Cl, I_pump


def v_dot(V, n, DKi, Kg, par):
    """dV/dt of the isolated neuron (no synaptic input)."""
    I_Na, I_K, I_Cl, I_pump = currents(V, n, DKi, Kg, par)
    return -(I_Na + I_K + I_Cl + I_pump) / par["Cm"]


def rhs(z, t, K_bath, par):
    V, n, DKi, Kg = z
    I_Na, I_K, I_Cl, I_pump = currents(V, n, DKi, Kg, par)
    K_o = P.concentrations(DKi, Kg, par)[3]
    dV = -(I_Na + I_K + I_Cl + I_pump) / par["Cm"]
    dn = (n_inf(V, par) - n) / par["tau_n"]
    dDKi = -(par["gamma"] / par["w_i"]) * (I_K - 2.0 * I_pump)
    dKg = par["epsilon"] * (K_bath - K_o)
    return [dV, dn, dDKi, dKg]


def simulate(K_bath, z0=(-15.0, 0.45, -3.5, -12.0), duration_ms=20_000.0,
             dt_ms=0.1, par=None):
    """Integrate the single neuron with LSODA (``scipy.integrate.odeint``).

    Returns a dict with keys t, V, n, DKi, Kg, Kext (all 1-D arrays sampled
    every ``dt_ms``).
    """
    par = P.DEFAULT if par is None else par
    t = np.arange(0.0, duration_ms + dt_ms / 2, dt_ms)
    z = odeint(rhs, list(z0), t, args=(float(K_bath), par))
    out = {"t": t, "V": z[:, 0], "n": z[:, 1], "DKi": z[:, 2], "Kg": z[:, 3]}
    out["Kext"] = P.k_ext(out["DKi"], out["Kg"], par)
    return out
