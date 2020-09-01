"""Test scattering resonances off of spherical wells."""
import wavefunctions as wfs

import functools

import numpy as np
from matplotlib import pyplot as plt

def V_sphere(x, V0, a, b):
    if a < x < a + b:
        return V0
    return 0


V_sphere = np.vectorize(V_sphere, otypes=[float])


def plot_cs():
    m = 1
    V0 = 1e2
    a = 1
    b = 0.09
    V = functools.partial(V_sphere, V0=V0, a=a, b=b)
    E = np.linspace(1, 100, 500)
    l = 0
    r_min = 1e-10
    r_max = 100
    resolution = 1e3
    tolerance = 1e-2

    cs = np.array([
        wfs.partial_cross_section(
            EE, V, m, l, r_max, r_min=r_min, resolution=resolution, 
            tolerance=tolerance
        ) for EE in E
    ])
    cs = cs * E / np.sin(wfs.E_to_k(E, m) * a) ** 2

    plt.plot(E, cs)
    # plt.savefig('hiiii.png')
    plt.show()


plot_cs()
