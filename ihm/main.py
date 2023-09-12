#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import klassez as kz

from spectra_reading import main as spectra_reading
from gen_param import main as gen_param
from fitting import main as fitting


# Load the acqus dictionary of the (mixture) spectrum
M = kz.Spectrum_1D('M.acqus', isexp=False)
M.fid = 7 * np.loadtxt('M.fid', dtype='complex128')
M.procs['zf'] = 16384
M.process()

acqus = M.acqus
N = M.r.shape[-1]

# Create list of peaks files
spectra_dir = ['C_1.fvf', 'C_2.fvf']


components = spectra_reading(M, spectra_dir)
#! until here correct

Lparam, param = gen_param(M, components)

fitting(M, len(components), Lparam, param)

