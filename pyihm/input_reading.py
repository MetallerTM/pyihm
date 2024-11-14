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

        # Processing options
        if 'PROC_OPTS' in lines[0]:
            lines.pop(0)    # Remove the header line
            dic['proc_opt'] = {}        # Placeholder
            for line in lines:
                line = line.split(':', 1)   # Separate at : to distinguish key and values
                if 'wf' in line[0]:         # Window function options
                    P = line[1].split(',')  # Separate additional arguments
                    # Add window function options, to resemble procs dictionary
                    dic['proc_opt']['wf'] = {}      # Placeholder
                    dic['proc_opt']['wf']['mode'] = f'{P[0]}'   
                    for PP in P:    # Other stuff
                        try:    # There might not be key=item things
                            key, value = PP.split('=')
                            dic['proc_opt']['wf'][f'{key}'] = eval(value)
                        except:
                            continue
                if 'zf' in line[0]:         # Zero-filling options
                    dic['proc_opt']['zf'] = int(eval(line[1]))
                if 'blp' in line[0]:        # Backward linear prediction
                    P = line[1].split(',')
                    dic['proc_opt']['blp'] = {} # Placeholder
                    for PP in P:    # Additional options for kz.processing.blp
                        try:
                            key, value = PP.split('=')
                            dic['proc_opt']['blp'][f'{key}'] = eval(value)
                        except:
                            continue
                if 'pknl' in line:          # PKNL yes or no
                    dic['proc_opt']['pknl'] = True
                if 'adjph' in line:         # Correct phase or not
                    dic['proc_opt']['adjph'] = True

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
                try:
                    dic['fit_lims'].append(tuple(eval(line)))
                except:
                    continue

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
            for il, l in enumerate(lines[1:]):
                line = l.split(',')  # Separates the various options
                dic['fit_kws'][il] = {} # Placeholder
                for kw in line:    # Loop on the parameters
                    if '=' not in kw:
                        continue
                    # Separate the key from the actual value
                    key, item = kw.split('=')
                    # Remove the spaces from the key
                    key = key.replace(' ', '')
                    try:
                        dic['fit_kws'][il][key] = eval(item)
                    except:
                        dic['fit_kws'][il][key] = f'{item}'.replace(' ', '')

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
        # Initial guess for the concentrations
        if 'I0' in lines[0]:
            dic['I0'] = eval('['+lines[1]+']')
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
            dic['proc_opt'],
    - comp_path: list
        Path to the .fvf files to be used for building the spectra of the components
    - fit_lims: tuple
        Limits of the fitting region, in ppm
    - fit_bds: dict
        Boundaries for the fitting parameters. The keywords are:
        > utol = allowed displacement for singlets and whole multiplets, in ppm 
        > utol_sg = allowed displacement for the peaks that are part of the same multiplet relatively to the center, in ppm
        > stol = allowed variation for the linewidth, in Hz 
        > ktol = allowed variation for the relative intensities within the same spectrum
    - fit_kws: list of dic
        Keyworded arguments for each run of the fit in custom mode (see lmfit)
    - plt_opt: dic
        Format and resolution of the figures: {'format'='tiff', 'dpi'=300}
    - Hs: list
        Nominal integrals for the components
    - I0: list or None
        Initial guess for the concentrations
    """
    # Get the dictionary of parameters
    dic = read_input_file(filename)

    # Check for missing entries
    if 'I0' not in dic.keys():
        dic['I0'] = None
    if 'mix_spectrum_txt' not in dic.keys():    # This is an optional parameter: replacement for spectrum
        dic['mix_spectrum_txt'] = None
    if 'fit_lims' not in dic.keys():
        print('Fit limits not found in the input file.')
        dic['fit_lims'] = None
    if 'fit_kws' not in dic.keys():    # This is an optional parameter: parameters for the fit routine
        dic['fit_kws'] = {}
    if len(dic['fit_kws'].keys()) == 0:    # If the dictionary is empty, it means that the user did not specify any parameter
        dic['fit_kws'] = {0: {}}    # Create a dummy dictionary
        dic['fit_kws'][0]['method'] = 'leastsq'
        dic['fit_kws'][0]['max_nfev'] = 10000
        dic['fit_kws'][0]['tol'] = 1e-5
    # make sure that max_nfev is an integer
    for key in dic['fit_kws'].keys():
        if 'max_nfev' in dic['fit_kws'][key].keys():
            dic['fit_kws'][key]['max_nfev'] = int(dic['fit_kws'][key]['max_nfev'])
        if dic['fit_kws'][key]['method'] == 'leastsq':
            if 'tol' not in dic['fit_kws'][key].keys():
                dic['fit_kws'][key]['tol'] = 1e-5
            if 'max_nfev' not in dic['fit_kws'][key].keys():
                dic['fit_kws'][key]['max_nfev'] = 10000
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
    for key, def_value in zip(['utol', 'utol_sg', 'stol', 'ktol'], [0.2, 0.1, 10, 0.01]):
        if key not in dic['fit_bds'].keys():    # Replace missing entries with default values
            dic['fit_bds'][key] = def_value

    # Add missing processing options with default values
    if 'proc_opt' not in dic.keys():    # Create it
        dic['proc_opt'] = {}
    if 'wf' not in dic['proc_opt'].keys():  # Set "no wf" as default
        dic['proc_opt']['wf'] = {'mode':'no'}
    if 'zf' not in dic['proc_opt'].keys():  # Do not zerofill
        dic['proc_opt']['zf'] = False
    if 'blp' not in dic['proc_opt'].keys(): # Do not perform blp
        dic['proc_opt']['blp'] = False
    if 'pknl' not in dic['proc_opt'].keys():    # Do not apply pknl
        dic['proc_opt']['pknl'] = False
    if 'adjph' not in dic['proc_opt'].keys():   # Do not phase
        dic['proc_opt']['adjph'] = False

    # Sort the values to be returned according to a meaningful scheme
    ret_vals = [
            dic['base_filename'],
            dic['mix_path'],
            dic['mix_kws'],
            dic['mix_spectrum_txt'],
            dic['proc_opt'],
            dic['comp_path'],
            dic['fit_lims'],
            dic['fit_bds'],
            dic['fit_kws'],
            dic['plt_opt'],
            dic['Hs'],
            dic['I0']
            ]
    return ret_vals


