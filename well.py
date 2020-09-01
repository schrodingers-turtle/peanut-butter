"""Test transmission resonances across 1D square wells.

hbar = 1.
"""
import wavefunctions as wfs

import functools

import numpy as np
import scipy.signal
from matplotlib import pyplot as plt

def V_well(x, V0, a, b):
    x0 = a / 2
    if x0 < abs(x) < x0 + b:
        return V0
    return 0


V_well = np.vectorize(V_well, otypes=[float])


m = 1
V0 = 100
a = 1
b = 1
E = np.linspace(0, 10, 100)
x = np.linspace(-(b + 1), b + 1, 20)
V = functools.partial(V_well, V0=V0, a=a, b=b)

def plot_T(V):
    T = [wfs.transmit_and_reflect(x, EE, V, m)[0] for EE in E]
    resonance_indices, _ = scipy.signal.find_peaks(T)
    resonances = E[resonance_indices]
    
    print('Resonances: E = {}'.format(resonances))
    plt.figure()
    plt.plot(E, T)
    plt.grid()
    plt.xlabel('$E$')
    plt.ylabel('$T$')


def plot_wf(E, V):
    T, R, t, r, wf = wfs.transmit_and_reflect(x, E, V, m, more=True)
    P = abs(wf) ** 2
    Vx = V(x)

    print('E = {}, T = {}'.format(E, T))
    plt.figure()
    plt.plot(x, Vx)
    plt.plot(x, P * max(Vx) / max(P))


plot_T(V)
plot_wf(4.79, V)
plt.show()
