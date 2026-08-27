"""Network of N all-to-all coupled HH-type neurons (Brian2), Eqs. 1 and 7.

Each neuron follows ``single_neuron.rhs`` plus a heterogeneous excitability
eta_i (Lorentzian, mean ``eta_bar``, half-width ``Delta``, truncated to
+-20) and instantaneous synapses: when neuron j spikes (V_j crosses ``Vth``)
every other neuron receives V_i += (J/N) (E - V_i).  ``J`` is the *total*
synaptic strength, as in the mean-field model.
"""
from __future__ import annotations

import numpy as np

from . import params as P

EQS = """
DNa_i = -DK_i : 1 (constant over dt)
DNa_o = -beta * DNa_i : 1 (constant over dt)
DK_o = -beta * DK_i : 1 (constant over dt)
K_i = K_i0 + DK_i : 1 (constant over dt)
Na_i = Na_i0 + DNa_i : 1 (constant over dt)
Na_o = Na_o0 + DNa_o : 1 (constant over dt)
K_o = K_o0 + DK_o + Kg : 1 (constant over dt)

n_inf = 1.0/(1.0+exp((Cnk-V)/DCnk)) : 1
m_inf = 1.0/(1.0+exp((Cmna-V)/DCmna)) : 1
h_n = 1.1 - 1.0/(1.0+exp(DChn*(n-Chn))) : 1

I_K    = (g_Kl+g_K*n)*(V - 26.64*log(K_o/K_i)) : 1
I_Na   = (g_Nal+g_Na*m_inf*h_n)*(V - 26.64*log(Na_o/Na_i)) : 1
I_Cl   = g_Cl*(V + 26.64*log(Cl_o0/Cl_i0)) : 1
I_pump = rho*(1.0/(1.0+exp((Cnap-Na_i)/DCnap))*(1.0/(1.0+exp((Ckp-K_o)/DCkp)))) : 1

dV/dt = (-1.0/Cm)*(I_Na+I_K+I_Cl+I_pump) + eta/tau_I : 1
dn/dt = (n_inf - n)/tau_n : 1
dDK_i/dt = -(gamma/w_i)*(I_K - 2*I_pump) : 1
dKg/dt = epsilon*(K_bath - K_o) : 1

Cnap : 1
DCnap : 1
Ckp : 1
DCkp : 1
Cmna : 1
DCmna : 1
Chn : 1
DChn : 1
Cnk : 1
DCnk : 1
Cm : second
tau_n : second
tau_I : second
g_Cl : 1
g_Na : 1
g_K : 1
g_Nal : 1
g_Kl : 1
w_i : 1
rho : 1
beta : 1
epsilon : hertz
gamma : hertz
K_bath : 1
Na_i0 : 1
Na_o0 : 1
K_i0 : 1
K_o0 : 1
Cl_o0 : 1
Cl_i0 : 1
E : 1
Jn : 1
eta : 1
"""


def lorentzian_eta(N, eta_bar, Delta, rng, cut=20.0):
    """Cauchy-distributed excitabilities truncated to eta_bar +- cut
    (values outside the window are replaced by eta_bar)."""
    s = rng.standard_cauchy(N) * Delta + eta_bar
    s[(s <= eta_bar - cut) | (s >= eta_bar + cut)] = eta_bar
    return s


def simulate(N=3000, K_bath=15.5, J=1.0, Delta=1.0, eta_bar=0.0, E=0.0,
             Vth=-50.0, duration_ms=20_000.0, dt_ms=0.01, record_dt_ms=1.0,
             z0=(-15.0, 0.45, -3.5, -12.0), par=None, E_i=None, seed=0,
             record_all=False, method="heun", report=None):
    """Simulate the population and return a dict.

    Keys: t, V_mean, n_mean, DKi_mean, Kg_mean, Kext_mean, rate (Hz),
    spikes (i, t) and, if ``record_all``, V and n as (N, time) arrays.
    ``E_i`` optionally overrides the reversal potential per neuron
    (Supplementary Fig. 1a uses E=-80 for a quarter of the neurons).
    """
    import brian2 as b2

    par = P.DEFAULT if par is None else par
    rng = np.random.default_rng(seed)
    b2.seed(seed)
    b2.start_scope()
    b2.defaultclock.dt = dt_ms * b2.ms

    G = b2.NeuronGroup(N, EQS, threshold="V > Vth", refractory="V > Vth",
                       method=method, namespace={"Vth": Vth})
    G.V = z0[0] + 1.0 * rng.standard_normal(N)
    G.n = z0[1] + 0.05 * rng.standard_normal(N)
    G.DK_i = z0[2] + 0.3 * rng.standard_normal(N)
    G.Kg = z0[3] + 0.3 * rng.standard_normal(N)
    for k in ("Cnap", "DCnap", "Ckp", "DCkp", "Cmna", "DCmna", "Chn", "DChn",
              "Cnk", "DCnk", "g_Cl", "g_Na", "g_K", "g_Nal", "g_Kl", "w_i", "rho",
              "Na_i0", "Na_o0", "K_i0", "K_o0", "Cl_o0", "Cl_i0"):
        setattr(G, k, par[k])
    G.beta = P.beta(par)
    G.Cm = par["Cm"] * b2.ms
    G.tau_n = par["tau_n"] * b2.ms
    G.tau_I = par["tau_I"] * b2.ms
    G.epsilon = par["epsilon"] / b2.ms
    G.gamma = par["gamma"] / b2.ms
    G.K_bath = K_bath
    G.E = E if E_i is None else np.asarray(E_i, float)
    G.Jn = J / N
    G.eta = lorentzian_eta(N, eta_bar, Delta, rng)

    S = b2.Synapses(G, G, on_pre="V += Jn*(E - V)")
    S.connect(condition="i != j")

    rec_vars = ("V", "n", "DK_i", "Kg")
    mon = b2.StateMonitor(G, rec_vars, record=True, dt=record_dt_ms * b2.ms)
    spk = b2.SpikeMonitor(G)
    rate = b2.PopulationRateMonitor(G)
    net = b2.Network(G, S, mon, spk, rate)
    net.run(duration_ms * b2.ms, report=report, report_period=30 * b2.second)

    V, n, DKi, Kg = (np.asarray(getattr(mon, v)) for v in rec_vars)
    out = {"t": np.asarray(mon.t / b2.ms), "V_mean": V.mean(0), "n_mean": n.mean(0),
           "DKi_mean": DKi.mean(0), "Kg_mean": Kg.mean(0),
           "V_var": V.var(0), "rate_t": np.asarray(rate.t / b2.ms),
           "rate": np.asarray(rate.rate / b2.Hz),
           "spikes_i": np.asarray(spk.i), "spikes_t": np.asarray(spk.t / b2.ms),
           "eta": np.asarray(G.eta)}
    out["Kext_mean"] = P.k_ext(out["DKi_mean"], out["Kg_mean"], par)
    if record_all:
        out["V"], out["n"] = V, n
    return out
