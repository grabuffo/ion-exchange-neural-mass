# TVB model class

`model_HH_ABH.py` is the mean-field model as a The Virtual Brain (TVB 2.x)
`Model` subclass, used for the connectome-based simulation of Fig. 6. This is
the corrected version: it includes the local recurrent term J r (E - V) of
Eq. 25 in dV/dt, which the version used at publication time omitted (the
effect on Fig. 6 is negligible, see ../REPRODUCTION_NOTES.md). The coupling
variable is x, hence the (R_minus/pi) factor on the coupling term.
`ionmf/network.py` is a dependency-free NumPy port of the same equations.
