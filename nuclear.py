"""Test resonanct scattering off of nuclear wells in Coulomb potentials.

In Hartree units.
"""
import wavefunctions as wfs

import numpy as np
from matplotlib import pyplot as plt
from scipy import constants

# Other units in terms of Hatree units.
m_proton = constants.physical_constants['proton-electron mass ratio'][0]
eV= constants.physical_constants['electron volt-hartree relationship'][0]
fermi = 1e-15 / constants.physical_constants['Bohr radius'][0]


def V_nuclear(r, R, V0=0, ZZ=1):
    """Coulomb potential with a flat well in the center.
    
    :param R: Nuclear radius / radius of the well.
    :param V0: Height of the inside of the well.
    :param ZZ: Product of the two charges.
    """
    if r <= R:
        return V0
    return ZZ / r


def V_proton_boron(r):
    return V_nuclear(r, 5 * fermi, -30e6 * eV, 5)


# V = lambda r: V_nuclear(r, 10, 0)
m = m_proton
l = 0
E = np.linspace(800, 803, 50) * eV
# E = np.linspace(1e-2, 0.5, int(1e3))
k = wfs.E_to_k(E, m)

# num = int(1e4)
# r = np.concatenate((
#     np.linspace(1e-2 * fermi, 1e2 * fermi, num), 
#     np.linspace(1e2 * fermi, 1e3 * fermi, num)[1:]
# ))

# psi, dpsi = wfs.energy_wf(r, (r[0], 1e4), 1e3 * eV, V_proton_boron, m)
# plt.plot(r[:num] / fermi, psi[:num])
# plt.show()

delta, _, r, psi = zip(*[
    wfs.phase_shift(
        E=EE, V=V_proton_boron, m=m, l=l, r_max=1e4, r_min=1e-2 * fermi, 
        resolution=1e5, more=True, verbose=True
    )
    for EE in E
])

plt.plot(E / eV, delta)
# plt.title(r'$\delta$ vs $E$ (natural units)')
plt.show()
plt.plot(E / eV, np.sin(delta) ** 2)
# plt.title(r'$\sin^2(\delta)$ vs $E$ (natural units)')
plt.show()

if 1:
    for EE, rr, psii in zip(E, r, psi):
        print(rr)
        print(f'E = {EE / eV} eV')
        plt.plot(rr, psii)
        plt.show()
