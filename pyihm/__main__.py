#! /usr/bin/env python3

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import klassez as kz

from .input_reading import read_input, select_regions
from .spectra_reading import main as spectra_reading
from .gen_param import main as gen_param
from .fit_mixture import main as do_fit

def print_header():
    print('*'*80)
    print('*'+' '*78+'*')
    print('*'+f'{"pyIHM":^78}'+'*')
    print('*'+f'{"-"*55:^78}'+'*')
    print('*'+f'{"Indirect Hard Modelling":^78}'+'*')
    print('*'+f'{"in Python":^78}'+'*')
    print('*'+' '*78+'*')
    print('*'*80)
    print()

print_header()

inp_files = sys.argv[1:]

for n_inp, inp_file in enumerate(inp_files):
    ## Read the input file to get the filenames and stuff
    print(f'pyIHM is now reading {inp_file} as {n_inp+1}/{len(sys.argv)-1} input file.\n')
    filename, mix_path, mix_kws, mix_txtf, comp_path, lims, bds, fit_kws, plt_opt = read_input(inp_file)


    ## Load the mixture spectrum
    print('Reading the mixture spectrum...')
    M = kz.Spectrum_1D(mix_path, **mix_kws)
    acqus = M.acqus
    # Do the FT
    M.process()
    # Replace the spectrum with one read from a text file, if told to do so
    if mix_txtf is not None:
        m_spect = np.loadtxt(mix_txtf, dtype='complex128')  # Always complex
        # Overwrite the complex, real and imaginary part of the spectrum
        M.S = m_spect
        M.r = m_spect.real
        M.i = m_spect.imag

    N = M.r.shape[-1]   # Number of points of the real part of the spectrum
    # Recompute the ppm scales to match the dimension of the spectrum
    M.freq = kz.processing.make_scale(N, acqus['dw'])
    M.ppm = kz.misc.freq2ppm(M.freq, acqus['SFO1'], acqus['o1p'])
    print(f'{os.path.join(M.datadir, M.filename)} successfully loaded.\n')

    # Sort the ppm limits so they appear always in the correct order
    if lims is None:    # Select them interactively
        lims = select_regions(M.ppm, M.r)
        text_to_append = '\n'.join([f'{max(X):-7.3f}, {min(X):-7.3f}' for X in lims])
        print('Append the following text to your input file:\n\nFIT_LIMITS\n'+text_to_append+'\n')
    lims = [(max(X), min(X)) for X in lims]

    ## Create list of peaks files
    print('Reading the pure components spectra...')
    components = spectra_reading(M, comp_path, lims)
    print(f'Done. {len(components)} spectra will be employed in the fit.\n')

    ## Create the parameters using lmfit
    print('Creating parameters for the fit...')
    Lparam, param = gen_param(M, components, bds)
    print('Done.\n')

    # Do the fit and save figures and output file
    do_fit(M, len(components), Lparam, param, lims, fit_kws, filename, **plt_opt)

    print('*'*80)
