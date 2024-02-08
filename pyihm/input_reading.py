#! /usr/bin/env python3

import sys
import os
import numpy as np
from .GUIs import select_regions

def read_input_file(filename):
    """
    Runs over the input file, looks for specific keywords, and interpret them accordingly.
    ----------
    Parameters:
    - filename: str
        Path to the input file
    ----------
    Returns: 
    - dic: dict
        Read values, organized
    """
    # Open the file and read the text
    f = open(filename, 'r')
    txt = f.read()

    # Check if all the mandatory sections are present: if not, raise an error
    mandatory_keys = [  # Keywords to look for
            'BASE_FILENAME',
            'MIX_PATH',
            'COMP_PATH',
            'FIT_BDS',
            ]
    exit_status = 0         # Missing keywords will make this go to 1
    for key in mandatory_keys:
        if key not in txt:
            print(f'ERROR: key "{key}" is missing!')
            exit_status = 1
    if exit_status:
        print('Aborting execution.\n')
        print('*'*80)
        exit()

    # Now, search the file for parameters. Break the text at empty lines
    blocks = txt.split(os.linesep+os.linesep)
    dic = {}    # Placeholder
    for block in blocks:
        lines = block.split(os.linesep)     # Equivalent to .readlines(), for each block

        # filename for the figures
        if 'BASE_FILENAME' in lines[0]:
            dic['base_filename'] = f'{lines[1]}'

        # Path to the mixture spectrum
        if 'MIX_PATH' in lines[0]:
            line = lines[1].split(',')  # Separates the actual path and the loading options
            dic['mix_path'] = line.pop(0)       # Store the filename
            dic['mix_kws'] = {}     # Placeholder
            # Loop on the remaining keywords
            for kw in line:
                # Skip empty lists to avoid stupid mistakes to stop the program
                if '=' not in kw:   
                    continue
                # Separate key from the value
                key, item = kw.split('=')
                key = key.replace(' ', '')  # Remove the spaces from the key
                try:    # If it is a number or something python can understand
                    dic['mix_kws'][key] = eval(item)
                except: # Store it as a string
                    dic['mix_kws'][key] = f'{item}'

        # Path to the spectrum to be overwritten
        if 'MIX_SPECTRUM_TXT' in lines[0]:
            dic['mix_spectrum_txt'] = lines[1]

        # Path to the components
        if 'COMP_PATH' in lines[0]:
            lines.pop(0)    # Remove the header line
            dic['comp_path'] = []   # Placeholder
            dic['Hs'] = []   # Placeholder
            for line in lines:  # One path per line
                if ',' in line:
                    nH = eval(line.rsplit(',')[-1].replace('H', ''))
                    line = line.rsplit(',')[0]
                else:
                    nH = 1
                dic['Hs'].append(nH)
                dic['comp_path'].append(line)

        # Delimiters of the fitting region, in ppm
        if 'FIT_LIMITS' in lines[0]:
            lines.pop(0)    # Remove the header line
            dic['fit_lims'] = []   # Placeholder
            for line in lines:  # One region per line
                dic['fit_lims'].append(tuple(eval(line)))

        # Boundaries for the parameters of the fit
        if 'FIT_BDS' in lines[0]:
            lines.pop(0)    # Remove header line
            dic['fit_bds'] = {} # Placeholder
            for kw in lines:    # Loop on the parameters
                if '=' not in kw:
                    continue
                # Separate the key from the actual value
                key, item = kw.split('=')
                # Remove the spaces from the key
                key = key.replace(' ', '')
                dic['fit_bds'][key] = eval(item)    # This is always a number

        if 'FIT_KWS' in lines[0]:
            dic['fit_kws'] = {} # Placeholder
            line = lines[1].split(',')  # Separates the various options
            for kw in line:    # Loop on the parameters
                if '=' not in kw:
                    continue
                # Separate the key from the actual value
                key, item = kw.split('=')
                # Remove the spaces from the key
                key = key.replace(' ', '')
                try:
                    dic['fit_kws'][key] = eval(item)
                except:
                    dic['fit_kws'][key] = f'{item}'.replace(' ', '')

        # Options for saving the figures: format and resolution
        if 'PLT_OPTS' in lines:
            # Same thing as before
            dic['plt_opt'] = {}
            for kw in lines[1].split(','):
                if '=' not in kw:
                    continue
                key, item = kw.split('=')
                key = key.replace(' ', '')
                try:
                    dic['plt_opt'][key] = eval(item)
                except:
                    dic['plt_opt'][key] = f'{item}'
    f.close()
    return dic


def read_input(filename):
    """
    Reads the input file to get all the information to perform the fit.
    The values read from the file are double-checked, and the missing entries are replaced with default values, so not to leave space to stupid mistakes.
    ---------
    Parameters:
    - filename: str
        Path to the input file
    ---------
    Returns:
    - base_filename: str
        Root of the name of all the files that the program will save
    - mix_path: str
        Path to the mixture spectrum
    - mix_kws: dict of keyworded arguments
        Additional instructions to be passed to kz.Spectrum_1D.__init__
    - mix_spectrum_txt: str or None
        Path to a .txt file that contains a replacement spectrum for the mixture
    - comp_path: list
        Path to the .fvf files to be used for building the spectra of the components
    - fit_lims: tuple
        Limits of the fitting region, in ppm
    - fit_bds: dict
        Boundaries for the fitting parameters. The keywords are:
        > utol = allowed displacement for singlets and whole multiplets, in ppm (absolute)
        > utol_sg = allowed displacement for the peaks that are part of the same multiplet relatively to the center, in ppm (absolute)
        > stol = allowed variation for the linewidth, in Hz (relative)
        > ktol = allowed variation for the relative intensities within the same spectrum(relative)
    """
    # Get the dictionary of parameters
    dic = read_input_file(filename)

    # Check for missing entries
    if 'mix_spectrum_txt' not in dic.keys():    # This is an optional parameter: replacement for spectrum
        dic['mix_spectrum_txt'] = None
    if 'fit_lims' not in dic.keys():
        print('Fit limits not found in the input file.')
        dic['fit_lims'] = None
    if 'fit_kws' not in dic.keys():    # This is an optional parameter: parameters for the fit routine
        dic['fit_kws'] = {}
    if 'method' not in dic['fit_kws'].keys():  # Algorithm to be used for the fit
        dic['fit_kws']['method'] = 'leastsq'
    if 'max_nfev' not in dic['fit_kws'].keys(): # Set default max_nfev
        dic['fit_kws']['max_nfev'] = 10000
    else:   # If it is set, make sure it is an integer
        dic['fit_kws']['max_nfev'] = int(dic['fit_kws']['max_nfev'])
    if 'tol' not in dic['fit_kws'].keys():      # Set default tolerance
        dic['fit_kws']['tol'] = 1e-5
    # Set the figures' options to .tiff 300 dpi, unless explicitely said
    if 'plt_opt' not in dic.keys():
        dic['plt_opt'] = {}
    if 'ext' not in dic['plt_opt'].keys():
        dic['plt_opt']['ext'] = 'tiff'
    if 'dpi' not in dic['plt_opt'].keys():
        dic['plt_opt']['dpi'] = 300
    else:
        # Make sure the resolution is an integer otherwise matplotlib gets offended
        dic['plt_opt']['dpi'] = int(dic['plt_opt']['dpi']) 

    # Double-check the boundaries of the fit
    for key, def_value in zip(['utol', 'utol_sg', 'stol', 'ktol'], [0.2, 0.01, 0.05, 0.01]):
        if key not in dic['fit_bds'].keys():    # Replace missing entries with default values
            dic['fit_bds'][key] = def_value


    # Sort the values to be returned according to a meaningful scheme
    ret_vals = [
            dic['base_filename'],
            dic['mix_path'],
            dic['mix_kws'],
            dic['mix_spectrum_txt'],
            dic['comp_path'],
            dic['fit_lims'],
            dic['fit_bds'],
            dic['fit_kws'],
            dic['plt_opt'],
            dic['Hs'],
            ]
    return ret_vals





