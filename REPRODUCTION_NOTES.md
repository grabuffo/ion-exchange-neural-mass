# Reproduction notes

This repository was assembled after publication from the authors' working
notebooks (`notebooks_original/`). Those notebooks depend on hard-coded local
and cluster paths, intermediate pickles that were not preserved, and an old
software stack, so the models and analyses were **re-implemented** in the
`ionmf` package and the `scripts/`. This file records what was checked, where
each parameter comes from, and the places where the new code differs from the
original one.

## Model equations

* `single_neuron.py`, `population.py` (Brian2 equation string) and
  `mean_field.py` are line-by-line transcriptions of the equations in the
  notebooks; `network.py` is a NumPy port of the TVB model class
  `notebooks_original/tvb_model/model_HH_ABH.py` (4th-order Runge-Kutta,
  dt = 0.1 ms, instantaneous coupling, state bound x >= 0, as in the
  original run).
* Membrane parameters are those of Table 1 of the paper, identical in every
  original notebook.
* The parabola coefficients printed by `fig2_single_neuron.py` (c- = -44.24,
  c+ = -20.44, R- = 0.336, R+ = -0.381, V* = -31.6) reproduce the published
  ones (-44.24, -20.40, 0.337, -0.381, -31) when the slow variables are
  frozen using the *whole* simulated trace (max of n, mean of DKi and Kg),
  which is what the original code did. Using only the late part of the
  trace changes the coefficients by roughly 10%.

## Where the per-figure parameters come from

Supplementary File 1 of the paper lists the parameters per figure, but its
column labels follow the *preprint* figure numbering. Mapping used here:

| Supplementary File 1 column | final figure | script |
|---|---|---|
| Fig. 2a | Fig. 2a | `fig2_single_neuron.py` |
| Fig. 3a (N = 3000, J = 1, R- = 0.34, R+ = -0.38) | Fig. 3 | `fig3_population_vs_meanfield.py` |
| Fig. 4a-b (Cm = 10, tau_n = 2.4, N = 3000) | Fig. 4b | `fig4_slow_potassium.py` |
| Fig. 4c (Cm = 16, tau_n = 8, gamma = 2.5e-4, eps = 1e-4, J = 0.01) | Fig. 4a | `fig4_slow_potassium.py` |
| Fig. 5c (J = 2, R- = 0.5, R+ = -0.5) | Fig. 4e | `fig4_slow_potassium.py` |
| Fig. 6b-c (J = 0.08, R- = 0.02, R+ = -0.1) | Fig. 6 | `fig6_network.py` |

Where the table and the notebooks disagree on initial conditions, the values
found in the notebook that produced the figure were used (e.g. Fig. 2a uses
z0 = (V, n, DKi, Kg) = (-15, 0.45, -3.5, -12), the notebook value).

## Known differences / decisions

1. **Fig. 2a, [K+]bath = 24.5.** The neuron is bistable at this value: from
   the common initial condition it spikes for ~2 s and then enters
   depolarization block; if [K+]ext starts at the bath value (Kg = 9.2) it
   stays in sustained spiking. The published panel shows sustained spiking,
   so the second initial condition is used for that panel only.
2. **Fig. 3b** compares each population with a mean-field run at the
   effective bath concentration [K+]bath^eff of Fig. 3c. Those values were
   obtained in the paper by visual inspection and are stored as data in the
   script (`KEFF`), not re-derived.
3. **Fig. 3a / Fig. 2-figure supplement 1a** histograms: the instants shown
   (one quiet, one inside a seizure-like event) are chosen automatically
   (last spike-free sample before the sample with most spiking neurons).
4. **Fig. 3-figure supplement 1** (per-neuron state classification): the
   code that produced the published panel was not archived. The script
   re-implements the procedure of the caption with an explicit rule
   (sliding 200 ms window on n_i: quiescent if max n < 0.15, depolarized if
   min n > 0.30, bursting otherwise; error = fraction of neurons whose state
   differs from that of the population mean). Numbers therefore differ in
   detail from the published ones (which report < 2% deviating neurons).
5. **Population simulations** use Brian2 with the Heun method, dt = 0.01 ms,
   spike threshold V > -50 mV and a fixed random seed. Original runs used
   dt = 0.01 ms and thresholds of -50/-40 mV depending on the notebook.
   Because the excitabilities eta_i are random, traces differ from the
   published ones in detail but not in regime.
6. **Fig. 4c-d** are experimental recordings; the plotting code expects csv
   files in `data/in_vitro/` (see its README).
7. **Fig. 5** is a numerical-continuation result and is not reproduced by
   simulation here (see `continuation/README.md`).
8. **Fig. 6** uses the same 6-node connectivity as the original run
   (`data/connectivity/6x6full.zip`, TVB format). The random weights of
   panel (a) are those of the file, not re-drawn.
