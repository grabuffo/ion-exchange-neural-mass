"""Network of mean-field nodes coupled through a structural connectome (Fig. 6).

This is a plain-numpy port of the TVB model class ``InfiniteHH`` used for the
paper (``notebooks_original/Jan/model_HH_ABH.py``): identical equations,
4th-order Runge-Kutta, instantaneous coupling (conduction speed = inf).

Node i receives the coupling term  (R_minus/pi) * G * sum_j W_ij x_j * (E - V_i)
added to dV_i/dt (TVB "Scaling" coupling on the state variable x).
"""
from __future__ import annotations

import numpy as np

from . import params as P
from .single_neuron import currents, n_inf


def dfun(state, W, G, K_bath, J, eta, Delta, coef, par, E=0.0):
    x, V, n, DKi, Kg = state
    I_Na, I_K, I_Cl, I_pump = currents(V, n, DKi, Kg, par)
    K_o = P.concentrations(DKi, Kg, par)[3]
    Vdot = -(I_Na + I_K + I_Cl + I_pump) / par["Cm"]
    coupling = G * (W @ x)
    r = coef["R_minus"] * x / np.pi

    lo = V <= coef["Vstar"]
    R = np.where(lo, coef["R_minus"], coef["R_plus"])
    c = np.where(lo, coef["c_minus"], coef["c_plus"])

    d = np.empty_like(state)
    d[0] = Delta + 2.0 * R * (V - c) * x - J * r * x
    d[1] = Vdot - R * x ** 2 + eta + (coef["R_minus"] / np.pi) * coupling * (E - V)
    d[2] = (n_inf(V, par) - n) / par["tau_n"]
    d[3] = -(par["gamma"] / par["w_i"]) * (I_K - 2.0 * I_pump)
    d[4] = par["epsilon"] * (K_bath - K_o)
    return d


def simulate(W, G, K_bath, J=0.08, eta=0.0, Delta=1.0, E=0.0, coef=None,
             par=None, z0=(0.1, -15.0, 0.45, -5.0, -16.0), duration_ms=20_000.0,
             dt_ms=0.1, record_every=1):
    """RK4 integration of ``len(W)`` coupled nodes.

    ``K_bath`` may be a scalar or one value per node. Returns dict with
    t, x, r [Hz], V, n, DKi, Kg (arrays of shape (time, nodes)).
    """
    par = P.DEFAULT if par is None else par
    coef = P.PARABOLA if coef is None else coef
    W = np.asarray(W, float)
    N = len(W)
    K_bath = np.broadcast_to(np.asarray(K_bath, float), (N,)).copy()
    args = (W, G, K_bath, J, eta, Delta, coef, par, E)

    nsteps = int(round(duration_ms / dt_ms))
    state = np.tile(np.asarray(z0, float)[:, None], (1, N))
    nrec = nsteps // record_every + 1
    rec = np.empty((nrec, 5, N))
    rec[0] = state
    k = 1
    for i in range(1, nsteps + 1):
        k1 = dfun(state, *args)
        k2 = dfun(state + 0.5 * dt_ms * k1, *args)
        k3 = dfun(state + 0.5 * dt_ms * k2, *args)
        k4 = dfun(state + dt_ms * k3, *args)
        state = state + dt_ms / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        state[0] = np.maximum(state[0], 0.0)  # TVB state bound x >= 0
        if i % record_every == 0:
            rec[k] = state
            k += 1
    rec = rec[:k]
    t = np.arange(k) * dt_ms * record_every
    return {"t": t, "x": rec[:, 0], "r": 1000.0 * coef["R_minus"] * rec[:, 0] / np.pi,
            "V": rec[:, 1], "n": rec[:, 2], "DKi": rec[:, 3], "Kg": rec[:, 4]}


def load_connectivity(path):
    """Load the 6-node TVB-style connectivity folder/zip used in Fig. 6.

    Returns (weights, tract_lengths, labels)."""
    import io
    import zipfile
    from pathlib import Path

    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            def read(name):
                cand = [n for n in zf.namelist() if n.endswith(name) and "__MACOSX" not in n]
                return io.BytesIO(zf.read(cand[0]))
            W = np.loadtxt(read("weights.txt"))
            L = np.loadtxt(read("tract_lengths.txt"))
            labels = [l.decode().strip() for l in read("region_labels.txt").read().splitlines()]
    else:
        W = np.loadtxt(path / "weights.txt")
        L = np.loadtxt(path / "tract_lengths.txt")
        labels = (path / "region_labels.txt").read_text().split()
    np.fill_diagonal(W, 0.0)
    return W, L, labels
