#! /usr/bin/env python3

"""
To run the program, type in the terminal:
$ python -m pyihm --input input_file1 input_file2 ... *flags

List of flags:
--debug: Print debug information
--cal: Perform calibration of the spectrum
--opt_method: Choose the type of conversion 
--input: Input files
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import klassez as kz
import argparse

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

## Parse the flags
#   Create the parser
parser = argparse.ArgumentParser(description='This program performs an Indirect Hard Modelling analysis on your spectra. For detailed info, please refer to the user manual in the "docs" folder of the GitHub repository: https://github.com/MetallerTM/pyihm')

#   Add the arguments
parser.add_argument('--input', nargs='+', help='Paths to input files')
parser.add_argument('--debug', action='store_true', help='Save figures of the ongoing fit every 20 iterations')
parser.add_argument('--cal', action='store_true', help='Perform calibration of the components before to start optimizations')
parser.add_argument('--opt_method', choices=['tight', 'fast', 'custom'], help='Choose the type of optimization for the core fit. DEFAULT: tight')
parser.add_argument('--rav', action='store_true', help='The GUIs will be drawn with a colorblind palette')
parser.add_argument('--noalgn', action='store_true', help='Skip the alignment fit of the components')


#   Parse the arguments
args = parser.parse_args()

#   Access your options as attributes of args
DEBUG_FLAG = args.debug
CAL_FLAG = args.cal
METHOD_FLAG = args.opt_method
RAV_FLAG = args.rav
NOALGN_FLAG = args.noalgn

#   Set 'tight' as default optimization method
if METHOD_FLAG is None:
    METHOD_FLAG = 'tight'
inp_files = args.input  # Access the input files through argparse

for n_inp, inp_file in enumerate(inp_files):
    ## Read the input file to get the filenames and stuff
    print(f'pyIHM is now reading {inp_file} as {n_inp+1}/{len(inp_files)} input file.\n')
    filename, mix_path, mix_kws, mix_txtf, proc_opt, comp_path, lims, bds, fit_kws, plt_opt, Hs, I0 = read_input(inp_file)

    # Create the folders where to save the data and the figures
    if os.sep in filename:
        base_dir, name = filename.rsplit(os.sep, 1)
    else:
        base_dir = os.getcwd()
        name = filename
    if f'{name}-FIGURES' not in os.listdir(base_dir):
        os.mkdir(os.path.join(base_dir, f'{name}-FIGURES'))
    if f'{name}-DATA' not in os.listdir(base_dir):
        os.mkdir(os.path.join(base_dir, f'{name}-DATA'))

    ## Load the mixture spectrum
    print('Reading the mixture spectrum...')
    M = kz.Spectrum_1D(mix_path, **mix_kws)
    acqus = M.acqus
    # Process
    if mix_txtf is None:
        for key, value in proc_opt['wf'].items():
            M.procs[key] = value
        if proc_opt['zf']:
            M.procs['zf'] = proc_opt['zf']
        if proc_opt['blp']:
            M.blp(**proc_opt['blp'])
        M.process()
        if proc_opt['pknl']:
            M.pknl()
        if proc_opt['adjph']:
            M.adjph()
    else:
        M.process()
        # Replace the spectrum with one read from a text file, if told to do so
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


    ## Create list of peaks files
    print('Reading the pure components spectra...')
    components, Hs, I0, lims, c_idx, I = spectra_reading(M, comp_path, Hs, lims, I0, CAL_FLAG, RAV_FLAG)
    print(f'Done. {len(components)} spectra will be employed in the fit.\n')

    ## Create the parameters using lmfit
    print('Creating parameters for the fit...')
    param = gen_param(M, components, bds, lims, Hs, c_idx, I0)
    print('Done.\n')

    # Do the fit and save figures and output file
    do_fit(M, len(components), Hs, param, I, lims, fit_kws, filename, NOALGN_FLAG, DEBUG_FLAG, METHOD_FLAG, **plt_opt)

    print('*'*80)

