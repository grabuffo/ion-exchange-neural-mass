# Biophysically inspired mean-field model of neuronal populations driven by ion exchange mechanisms

Code and data accompanying

> Rabuffo G, Bandyopadhyay A, Calabrese C, Gudibanda K, Depannemaecker D,
> Mitiko Takarabe L, Chatterjee S, Saggio ML, Desroches M, Ivanov A, Linne M-L,
> Bernard C, Petkoski S, Jirsa VK. *Biophysically inspired mean-field model of
> neuronal populations driven by ion exchange mechanisms.* eLife (2026).
> https://doi.org/10.7554/eLife.104071

The paper derives a 5-dimensional neural mass model from a network of
Hodgkin-Huxley-type neurons whose excitability is controlled by ion exchange
with the extracellular space and an external potassium bath. This repository
contains a small, dependency-light Python package with the four models used in
the paper and one script per figure that re-simulates everything from scratch.

## Contents

```
ionmf/                 the models
  params.py            biophysical parameters (Table 1) and nullcline coefficients
  single_neuron.py     single HH-type neuron with ion dynamics        (Eq. 1)
  population.py        N all-to-all coupled neurons, Brian2           (Eqs. 1, 7)
  mean_field.py        5-D mean-field / neural mass model             (Eq. 25)
  network.py           mean-field nodes coupled through a connectome  (Fig. 6)
  nullcline.py         dV/dt geometry and the two-parabola fit        (Fig. 2c-d)
scripts/               one script per figure (see table below)
figures/               output of the scripts (pdf + png)
data/connectivity/     6-node connectome used in Fig. 6
data/in_vitro/         in vitro LFP + [K+]ext recordings of Fig. 4c-d (csv; being added, see below)
continuation/          numerical-continuation files behind Fig. 5 (being added, see below)
tvb/                   the model as a TVB Model class (used for Fig. 6)
REPRODUCTION_NOTES.md  what was re-implemented, parameter sources, known differences
```

## Data availability (in progress)

Everything needed to re-simulate the model panels is in this repository. Two
sets of files are still being collected from the co-authors who produced them
and will be added here:

* the in vitro LFP and [K+]ext recordings shown in Fig. 4c-d (`data/in_vitro/`);
* the numerical-continuation files behind the bifurcation diagram of Fig. 5
  (`continuation/`).

Until then, they are available on request from the corresponding author,
Giovanni Rabuffo (giovanni.rabuffo@univ-amu.fr).

## Installation

```bash
git clone https://github.com/grabuffo/ion-exchange-neural-mass.git
cd ion-exchange-neural-mass
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # numpy, scipy, matplotlib, brian2, networkx
```

Tested with Python 3.11-3.14, NumPy 2.x, SciPy 1.17, Brian2 2.10.

## Reproducing the figures

```bash
./run_all.sh            # everything, ~1-2 h on a laptop (population runs dominate)
./run_all.sh --quick    # ~5 min smoke test with tiny networks (not quantitative)
```

or run the scripts individually from `scripts/`:

| script | figure | what is simulated | runtime |
|---|---|---|---|
| `fig2_single_neuron.py` | Fig. 2a-d | single neuron for five [K+]bath values; dV/dt curves for 25 [K+]bath values and the parabola fit (prints c-, c+, R-, R+, V*) | 3 min |
| `fig3_population_vs_meanfield.py` | Fig. 3a-c | six networks of N = 3000 neurons (24 s each) and the matching mean-field runs | 30-60 min |
| `fig4_slow_potassium.py` | Fig. 4a, b, e (+ c, d if data present) | mean field with slow ion dynamics (15 min simulated), N = 3000 network with rescaled parameters, emergent bursting regime | 15 min |
| `fig6_network.py` | Fig. 6a-c | six coupled neural masses, G = 0 and G = 100 | 1 min |
| `figS1_bimodal_and_psd.py` | Fig. 2-figure supplement 1 | N = 1000 network with a bimodal V distribution; 4 x 31 mean-field runs for the PSD sweeps | 20 min |
| `figS2_first_moment_error.py` | Fig. 3-figure supplement 1 | seven N = 3000 networks, per-neuron state classification | 45-90 min |

Fig. 1 is a schematic. Fig. 5 (two-parameter bifurcation diagram of the fast
subsystem) was obtained by numerical continuation, see `continuation/`.

Simulation results are cached in `cache/` (created on first run) so that the
scripts can be re-run to change the plots without re-simulating. Population
scripts accept `--N`, `--duration` and `--workers` (parallel Brian2 runs).

## Using the models

```python
from ionmf import single_neuron, mean_field, population, network, params

# single neuron, seizure-like events
s = single_neuron.simulate(K_bath=15.5, duration_ms=20_000)      # s["V"], s["Kext"], ...

# mean field at the same bath concentration (J = 1, Delta = 1)
m = mean_field.simulate(K_bath=15.1, J=1.0, Delta=1.0)           # m["V"], m["r"] (Hz), ...

# 3000 coupled neurons (Brian2)
p = population.simulate(N=3000, K_bath=15.5, J=1.0, Delta=1.0)   # p["V_mean"], p["spikes_t"], ...

# six neural masses on the Fig. 6 connectome
W, L, labels = network.load_connectivity("data/connectivity/6x6full.zip")
n = network.simulate(W, G=100.0, K_bath=[5.5, 5.5, 5.5, 15.5, 5.5, 5.5], J=0.08,
                     coef=params.parabola(R_minus=0.02, R_plus=-0.1))
```

All parameters default to Table 1 of the paper; `params.make(Cm=16, tau_n=8)`
and `params.parabola(R_minus=0.5, R_plus=-0.5)` create modified sets. The
per-figure parameter values are listed in Supplementary File 1 of the paper and
hard-coded in the corresponding script.

## License

MIT (see `LICENSE`). If you use this code please cite the paper above.
