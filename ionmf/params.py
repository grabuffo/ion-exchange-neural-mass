"""Default parameters of the model (Table 1 of the paper).

All conductances are in nS, concentrations in mM (mol m^-3), voltages in mV,
times in ms. ``26.64`` mV is the thermal voltage RT/F used in the Nernst terms.
"""
from __future__ import annotations

import copy

# Biophysical parameters shared by the single neuron, the population and the
# mean-field model (Table 1). The four "time" parameters (Cm, tau_n, gamma,
# epsilon) are the ones that are re-scaled in Fig. 4 (see Supplementary File 1).
DEFAULT = {
    "Cnap": 21.0, "DCnap": 2.0,      # Na+ pump activation (mM)
    "Ckp": 5.5, "DCkp": 1.0,         # K+ pump activation (mM)
    "Cmna": -24.0, "DCmna": 12.0,    # m_inf sigmoid (mV)
    "Chn": 0.4, "DChn": -8.0,        # h(n) sigmoid
    "Cnk": -19.0, "DCnk": 18.0,      # n_inf sigmoid (mV)
    "g_Cl": 7.5, "g_Na": 40.0, "g_K": 22.0, "g_Nal": 0.02, "g_Kl": 0.12,  # nS
    "rho": 250.0,                    # pump strength (pA)
    "w_i": 2160.0, "w_o": 720.0,     # intra/extracellular volumes (um^3)
    "Na_i0": 16.0, "Na_o0": 138.0, "K_i0": 130.0, "K_o0": 4.80,
    "Cl_o0": 112.0, "Cl_i0": 5.0,
    "Cm": 1.0, "tau_n": 4.0, "gamma": 0.04, "epsilon": 0.001, "tau_I": 1.0,
}

# Piecewise-quadratic approximation of the V-nullcline, fitted in
# ``nullcline.fit_parabolas`` (Fig. 2c-d, Supplementary File 1, Fig. 3 column).
PARABOLA = {"c_minus": -44.240, "c_plus": -20.400,
            "R_minus": 0.337, "R_plus": -0.381, "Vstar": -31.0}

THERMAL_V = 26.64  # mV


def make(**overrides) -> dict:
    """Return a copy of ``DEFAULT`` with ``overrides`` applied."""
    p = copy.deepcopy(DEFAULT)
    unknown = set(overrides) - set(p)
    if unknown:
        raise KeyError(f"unknown parameter(s): {sorted(unknown)}")
    p.update(overrides)
    return p


def parabola(**overrides) -> dict:
    p = dict(PARABOLA)
    unknown = set(overrides) - set(p)
    if unknown:
        raise KeyError(f"unknown coefficient(s): {sorted(unknown)}")
    p.update(overrides)
    return p


def beta(par: dict) -> float:
    """Volume ratio w_i / w_o."""
    return par["w_i"] / par["w_o"]


def concentrations(DKi, Kg, par: dict):
    """Ion concentrations from the two slow variables (Eqs. 3-6)."""
    b = beta(par)
    K_i = par["K_i0"] + DKi
    Na_i = par["Na_i0"] - DKi
    Na_o = par["Na_o0"] + b * DKi
    K_o = par["K_o0"] - b * DKi + Kg
    return K_i, Na_i, Na_o, K_o


def k_ext(DKi, Kg, par: dict):
    """Extracellular potassium concentration [K+]_ext (= K_o)."""
    return par["K_o0"] - beta(par) * DKi + Kg
