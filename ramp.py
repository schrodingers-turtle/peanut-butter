"""Test transmission across linear ramps of different widths."""
import wavefunctions as wfs

import functools

import numpy as np
from matplotlib import pyplot as plt

def V_ramp(x, a, V0):
    if x <= 0:
        return 0
    elif x >= a:
        return V0
    else:
        return V0 * x / a


V_ramp = np.vectorize(V_ramp, otypes=[float])


m = 1
E = 1
V0 = 0.8
a = np.linspace(0.01, 10, 20)
x = np.linspace(0, max(a), 50)
V = [functools.partial(V_ramp, a=aa, V0=V0) for aa in a]
T, R = np.array([wfs.transmit_and_reflect(x, E, VV, m) for VV in V]).T
print(a)
print(T)
if 0:
    for VV in V:
        plt.figure()
        plt.plot(x, VV(x))
plt.show()
