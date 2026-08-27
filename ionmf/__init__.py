"""ionmf -- biophysically inspired mean-field model of neuronal populations
driven by ion-exchange mechanisms (Rabuffo et al., eLife).

Modules
-------
params        : default biophysical parameters and nullcline (parabola) coefficients
single_neuron : Hodgkin-Huxley-type single neuron with ion dynamics (Eq. 1 of the paper)
population    : network of N all-to-all coupled HH-type neurons (Brian2)
mean_field    : 5-D mean-field (neural mass) model (Eq. 25 of the paper)
network       : several mean-field nodes coupled through a structural connectome (Fig. 6)
nullcline     : V-nullcline geometry and the piecewise-quadratic (parabola) fit
plotting      : small helpers for the figure scripts
"""
from . import params, single_neuron, mean_field, network, nullcline, plotting  # noqa: F401

__version__ = "1.0.0"
