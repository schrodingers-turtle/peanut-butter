"""Misc short tests."""
import wavefunctions as wfs

import multiprocessing

import numpy as np
from matplotlib import pyplot as plt

m = 1


def free_space_cs_test():
    E = np.linspace(1, 10, 10)
    V = lambda r: 0

    cs_result = np.array([wfs.partial_cross_section(
        EE, V, m, 2, 50, r_min=1e-3, use_scipy=False, more=True
    ) for EE in E], dtype=object)
    print(cs_result[:, :2])


greg = None
def pool_test():
    """Test multiprocessing."""
    global greg
    def greg_(x):
        return 2 * x
    greg = greg_
    with multiprocessing.Pool(2) as pool:
        return pool.map(greg, [1, 2, 3]), pool


if __name__ == '__main__':
    multiprocessing.freeze_support()

    # free_space_cs_test()
    print(pool_test())

