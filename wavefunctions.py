"""A module for integrating the Schrodinger equation, finding
transmission probabilities and scattering cross sections, etc.
"""
import functools
import multiprocessing
import warnings

import scipy.constants, scipy.integrate, scipy.signal
import numpy as np

hbar = 1

# remove later
constants = scipy.constants
eV= constants.physical_constants['electron volt-hartree relationship'][0]
fermi = 1e-15 / constants.physical_constants['Bohr radius'][0]


def energy_wf(x, psii0, E, V, m):
    """Find a definite energy wavefunction by integrating the 
    Schrodinger equation.

    :return: (wavefunction, its derivative)
    """
    if len(psii0) not in (2, 4):
        raise ValueError
    is_real = len(psii0) == 2

    if is_real:
        def dpsii(psii, x):
            factor = 2 * m * (V(x) - E) / hbar ** 2
            return psii[1],  factor * psii[0]
    else:
        def dpsii(psii, x):
            factor = 2 * m * (V(x) - E) / hbar ** 2
            return psii[2], psii[3], factor * psii[0], factor * psii[1]
    
    result = scipy.integrate.odeint(dpsii, psii0, x)

    if is_real:
        return result.T
    else:
        wf = result[:, 0] + 1j * result[:, 1]
        dwf = result[:, 2] + 1j * result[:, 3]
        return wf, dwf


def transmit_and_reflect(x, E, V, m, V_in=None, V_out=None, more=False):
    """Find transmission and reflection amplitudes."""
    x_in, x_out = x[0], x[-1]
    if V_in is None:
        V_in = V(x_in)
    if V_out is None:
        V_out = V(x_out)
    
    source_direction = 1 if x_out > x_in else -1
    k_in, k_out = (source_direction * E_to_k(E - VV, m) for VV in (V_in, V_out))
    psii0 = (1, 0, 0, k_out)
    psi, dpsi = energy_wf(x[::-1], psii0, E, V, m)
    psi, dpsi = psi[::-1], dpsi[::-1]

    psi_in, dpsi_in = psi[0], dpsi[0]
    i = (psi_in - 1j * dpsi_in / k_in) / 2
    r = (psi_in + 1j * dpsi_in / k_in) / 2
    t, r = 1 / i, r / i
    T, R = abs(t) ** 2 * k_out / k_in, abs(r) ** 2

    if more:
        return T, R, t, r, psi
    return T, R


def partial_cross_section(E, V, m, l, r_max, more=False, **kwargs):
    """Find a partial cross section for a radial potential."""
    phase_shift_result = phase_shift(E, V, m, l, r_max, more=more, **kwargs)
    if more:
        delta, *results = phase_shift_result
    else:
        delta = phase_shift_result
    
    k = E_to_k(E, m)
    cross_section = delta_to_cs(delta, k, l)

    if more:
        return cross_section, delta, *results
    return cross_section


def phase_shift(
    E, V, m, l, r_max, r_min=0, tolerance=1e-2, resolution=20, 
    num_consistent_peaks=3, use_scipy=False, more=False, verbose=False
):
    """Find a phase shift of an energy eigenstate of a radial
    potential.
    
    :param num_consistent_peaks: Require this many final peaks that do
        not differ within the tolerance range.
    :param use_scipy: If True, use SciPy's `find_peaks`.
    """
    if E == 0:
        raise ZeroDivisionError(
            "Can't find scattering results for a wave with zero energy."
        )
    if l > 0 and r_min == 0:
        raise ZeroDivisionError(
            "V_eff blows up at the origin for l > 0. Choose an "
            "r_min larger than 0."
        )
    if verbose:
        print('Starting phase shift for E = {}, l = {}.'.format(E, l))

    V_eff = V_to_V_eff(V, l, m)
    k = E_to_k(E, m)
    wavelength = 2 * np.pi / k
    r_tol = tolerance * wavelength
    dr = wavelength / resolution

    # move window out and find peaks
    i = 0
    within_tol = False
    past_r_max = False
    r = [r_min]
    psi = [r_min]
    dpsi = [1]
    r_peaks = np.array([])
    while not (within_tol or past_r_max):
        r, psi, dpsi, r_peaks, past_r_max, within_tol, i = _do_window(
            r, psi, dpsi, r_peaks, past_r_max, within_tol, i, r_min, r_max, dr, 
            E, V_eff, m, wavelength, num_consistent_peaks, r_tol, use_scipy
        )
    
    if r_peaks.size == 0:
        raise RuntimeError("No peaks found.")
    if not within_tol:
        raise RuntimeError(
            "Max r reached. If `r_max` is sufficiently large, then try "
            "increasing the resolution or decreasing the tolerance."
        )

    delta = - (k * r_peaks[-1] - (l + 1) * np.pi / 2)

    if more:
        return delta, V_eff, r, psi
    return delta


def _do_window(
    r, psi, dpsi, r_peaks, past_r_max, within_tol, i, r_min, r_max, dr, E, 
    V_eff, m, wavelength, num_consistent_peaks, r_tol, use_scipy
):
    def returnn():
        nonlocal i
        i += 1
        return r, psi, dpsi, r_peaks, past_r_max, within_tol, i
    
    r, past_r_max, r_window = _move_r(
        r, past_r_max, r_min, r_max, dr, i, wavelength)
    psi, dpsi, psi_window, dpsi_window = _move_psi(
        psi, dpsi, r_window, E, V_eff, m)
    
    if use_scipy:
        peak_indices, _ = scipy.signal.find_peaks(psi_window)
    else:
        peak_indices = zero_crossings(dpsi_window)
    if peak_indices.size == 0:
        return returnn()

    # add new peaks
    r_peaks_window = r_window[peak_indices] - i * wavelength
    r_peaks = np.concatenate((r_peaks, r_peaks_window))

    if len(r_peaks) < num_consistent_peaks:
        return returnn()

    # check peaks
    if (np.abs(np.diff(r_peaks[-num_consistent_peaks:])) < r_tol).all():
        within_tol = True
    return returnn()


def _move_r(r, past_r_max, r_min, r_max, dr, i, wavelength):
    """Move r to a new window."""
    if i == 0:
        r_window = arange_filled(r_min, wavelength, dr)
    else:
        r_window = arange_filled(i * wavelength, (i + 1) * wavelength, dr)
    r = np.concatenate((r, r_window[1:]))
    if r_window[-1] >= r_max:
        past_r_max = True
    return r, past_r_max, r_window


def _move_psi(psi, dpsi, r_window, E, V_eff, m):
    """Move the wavefunction to a new window."""
    psii0 = psi[-1], dpsi[-1]
    psi_window, dpsi_window = energy_wf(r_window, psii0, E, V_eff, m)
    psi = np.concatenate((psi, psi_window[1:]))
    dpsi = np.concatenate((dpsi, dpsi_window[1:]))
    return psi, dpsi, psi_window, dpsi_window


def zero_crossings(x, direction=-1):
    """Find the indices where `x` crosses zero in the given direction.

    Has certain shortcomings (when `x` lingers at 0), but works for
    finding phase shifts.
    """
    sign = np.sign(x)
    diff = np.diff(sign)
    return np.where(
        (diff == 2 * direction) | ((diff == direction) & (sign[1:] == 0))
    )[0] + 1


def delta_to_cs(delta, k, l):
    """Convert a phase shift delta to a partial cross section."""
    return 4 * np.pi * (2 * l + 1) * np.sin(delta) ** 2 / k ** 2



def E_to_k(E, m):
    """Convert an energy to a wavevector."""
    return np.sqrt(2 * m * E) / hbar


def k_to_E(k, m):
    """Convert a wavector to an energy."""
    return (hbar * k) ** 2 / (2 * m)


def V_to_V_eff(V, l, m):
    """Convert a potential energy function to an effective potential
    energy function."""
    if l == 0:
        return V
    
    @functools.wraps(V)
    def V_eff(r, *args, **kwargs):
        return V(r, *args, **kwargs) + \
            hbar ** 2 * l * (l + 1) / (2 * m * r ** 2)

    return V_eff


def arange_filled(start, stop, step, *args, shrink=True, **kwargs):
    """Like `np.arange`, but shrink `step` so that the sequence reaches
    `stop` exactly, and include `stop`.
    """
    range_ = stop - start
    num_steps = range_ / step
    if not num_steps.is_integer():
        step = range_ / np.ceil(num_steps)
    return np.arange(start, stop + step / 2, step, *args, **kwargs)


def parallelize(func):
    """Make `func` run in parallel on multiple processors."""
    @functools.wraps(func)
    def parallelized(x, *args, **kwargs):
        with multiprocessing.Pool(2) as pool:
            return


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    m = 1

    E = 1
    V = lambda x: 0.0 if x < 5 else 0

    cs, delta, V_eff, r, psi = partial_cross_section(
        E, V, m, 2, 50, r_min=1e-3, more=True)
    print(cs, delta)
    plt.plot(r, psi)
    plt.savefig('hi.png')
    plt.show()
