#! /usr/bin/env python3

import os
from datetime import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt
import klassez as kz
import lmfit as l

from .gen_param import main as gen_param
from .gen_param import L2P
from . import plots


def calc_spectra(Lparam, param, N_spectra, acqus, N):
    """
    Computes the spectra to be used as components for the fitting procedure, in form of lists of 1darrays. Each array is the sum of all the peaks.
    This function is called at each iteration of the fit.
    ---------
    Parameters:
    - Lparam: lmfit.Parameters object
        Normalized parameters
    - param: lmfit.Parameters object
        Actual parameters
    - N_spectra: int
        Number of spectra to be used as components
    - acqus: dict
        Dictionary of acquisition parameters
    - N: int
        Number of points for zero-filling, i.e. final dimension of the arrays
    ---------------
    Returns:
    - spectra: list of 1darray
        Computed components of the mixture, weighted for their relative intensity
    """
    # Convert normalized parameters into new parameters
    for key in Lparam:
        if key == 'count':
            continue
        newkey = key.replace('L', '', 1)
        value = L2P(Lparam[key].value, param[newkey].min, param[newkey].max)
        param[newkey].set(value)

    # Separate the parameters according to the spectra
    d_param = param.valuesdict()    # Convert to dictionary
    keys = list(d_param.keys())     # Get the keys
    spectra_par = []    # Placeholder

    for n in range(N_spectra):  # read: for each spectrum
        # Make a list of dictionaries. Each dictionary contains only the parameters relative to a given spectrum
        spectra_par.append({key.replace(f'S{n+1}_', ''): d_param[key] for key in keys if f'S{n+1}' in key})

    # How many peaks there are in each spectrum?
    peaks_idx = []  # Placeholder
    for par in spectra_par: # read: for each spectrum
        # Get the indices of the peaks 
        idxs = [eval(key.replace(f'u', '')) for key in par if 'u' in key]
        # Append it in the list
        peaks_idx.append(idxs)

    # Now we make the spectra!
    spectra = []    # Placeholder
    for n in range(N_spectra):
        dic = dict(spectra_par[n])  # Alias for the n-th spectrum peaks
        # Generate the fit.Peak objects
        peak_list = [kz.fit.Peak(acqus,
            u=dic[f'u{i}'], 
            fwhm=dic[f's{i}'],
            k=dic[f'k{i}'],
            x_g=0, 
            phi=0,
            N=N,
            ) for i in peaks_idx[n]]
        # Compute the trace for each peak, then sum them up, finally multiply by the intensity
        spectra.append(dic['I'] * np.sum([peak() for peak in peak_list], axis=0))
    return spectra

def calc_spectra_obj(Lparam, param, N_spectra, acqus, N):
    """
    Computes the spectra to be used as components for the fitting procedure, in form of lists of kz.fit.Peak objects. 
    ---------
    Parameters:
    - Lparam: lmfit.Parameters object
        Normalized parameters
    - param: lmfit.Parameters object
        Actual parameters
    - N_spectra: int
        Number of spectra to be used as components
    - acqus: dict
        Dictionary of acquisition parameters
    - N: int
        Number of points for zero-filling, i.e. final dimension of the arrays
    ---------------
    Returns:
    - spectra: list of kz.fit.Peak objects
        Computed components of the mixture, weighted for their relative intensity
    """
    # Convert normalized parameters into new parameters
    for key in Lparam:
        if key == 'count':
            continue
        newkey = key.replace('L', '', 1)
        value = L2P(Lparam[key].value, param[newkey].min, param[newkey].max)
        param[newkey].set(value)

    # Separate the parameters according to the spectra
    d_param = param.valuesdict()    # Convert to dictionary
    keys = list(d_param.keys())     # Get the keys
    spectra_par = []    # Placeholder

    for n in range(N_spectra):  # read: for each spectrum
        # Make a list of dictionaries. Each dictionary contains only the parameters relative to a given spectrum
        spectra_par.append({key.replace(f'S{n+1}_', ''): d_param[key] for key in keys if f'S{n+1}' in key})

    # How many peaks there are in each spectrum?
    peaks_idx = []  # Placeholder
    for par in spectra_par: # read: for each spectrum
        # Get the indices of the peaks 
        idxs = [eval(key.replace(f'u', '')) for key in par if 'u' in key]
        # Append it in the list
        peaks_idx.append(idxs)

    # Now we make the spectra!
    spectra = []    # Placeholder
    for n in range(N_spectra):
        dic = dict(spectra_par[n])  # Alias for the n-th spectrum peaks
        # Generate the fit.Peak objects
        peak_list = [kz.fit.Peak(acqus,
            u=dic[f'u{i}'], 
            fwhm=dic[f's{i}'],
            k=dic[f'k{i}'],
            x_g=dic[f'x_g{i}'], 
            phi=0,
            N=N,
            ) for i in peaks_idx[n]]
        # Add the list of peaks to the final list
        spectra.append(peak_list)
    return spectra

def f2min(Lparam, param, N_spectra, acqus, N, exp, I, plims, cnvg_path):
    """
    Function to compute the quantity to be minimized by the fit.
    ----------
    Parameters:
    - Lparam: lmfit.Parameters object
        Normalized parameters
    - param: lmfit.Parameters object
        actual parameters
    - N_spectra: int
        Number of spectra to be used as components
    - acqus: dict
        Dictionary of acquisition parameters
    - N: int
        Number of points for zero-filling, i.e. final dimension of the arrays
    - exp: 1darray
        Experimental spectrum
    - I: float
        Intensity correction for the calculated spectrum. Used to maintain the relative intensity small.
    - plims: slice
        Delimiters for the fitting region. The residuals are computed only in this regio. They must be given as point indices
    - cnvg_path: str
        Path for the file where to save the convergence path
    ----------
    Returns:
    - target: float
        \sum [ (exp - I*calc)^2 ]
    """
    param['count'].value += 1
    count = param['count'].value
    # Compute the trace for each spectrum
    spectra = calc_spectra(Lparam, param, N_spectra, acqus, N)
    spectra_T = [np.concatenate([spectrum[w] for w in plims]) for spectrum in spectra]

    # Sum the spectra to give the total fitting trace
    total = np.sum(spectra_T, axis=0)
    # Make the residuals
    residual = exp/I - total
    t_residual = residual
    target = np.sum(t_residual**2) / len(t_residual)
    # Print how the fit is going, both in the file and in standart output
    with open(cnvg_path, 'a', buffering=1) as cnvg:
        cnvg.write(f'{count:5.0f}\t{target:10.5e}\n')
    print(f'Iteration step: {count:5.0f}; Target: {target:10.5e}', end='\r')
    return target

def write_output(M, I, K, spectra, lims, filename='fit.report'):
    """
    Write a report of the performed fit in a file.
    The parameters of the single peaks are saved using the kz.fit.write_vf function.
    -----------
    Parameters:
    - M: kz.Spectrum_1D object
        Mixture spectrum
    - I: float
        Absolute intensity for the calculated spectrum
    - K: sequence
        Relative intensities of the spectra in the mixture
    - lims: tuple
        Boundaries of the fit region
    - filename: str
        Name of the file where to write the files.
    """
    # Get missing information
    N_spectra = len(spectra)    # Number of spectra
    now = datetime.now()
    date_and_time = now.strftime("%d/%m/%Y at %H:%M:%S")

    # Erase previously present file
    f = open(filename, 'w', buffering=1)

    ## HEADER
    f.write('Fit performed by {} on {}\n\n'.format(os.getlogin(), date_and_time))
    f.write(f'Mixture spectrum: {os.path.join(M.datadir, M.filename)}\n\n')
    f.write(f'Absolute intensity correction: I = {I:.5e}\n\n')
    f.write('Relative intensities:\n')
    for k, r_i in enumerate(K):
        f.write(f'Comp. {k+1:>3}: {r_i:.5f}\n')
    f.write('\n\n\n')
    f.close()

    ## PARAMETERS OF THE PEAKS
    for k, component in enumerate(spectra):
        # Make a dictionary of peak objects in order to use kz.fit.write_vf
        dict_component = {j+1: peak for j, peak in enumerate(component)}
        # Spacer for the spectrum identifier
        with open(filename, 'a', buffering=1) as f:
            f.write(f'Component {k+1} fitted parameters:\n')
        # Do the writing
        kz.fit.write_vf(filename, dict_component, lims, K[k]*I)
        # Add space
        with open(filename, 'a', buffering=1) as f:
            f.write(f'\n\n')
        

def main(M, N_spectra, Lparam, param, lims=None, fit_kws={}, filename='fit', ext='tiff', dpi=600):
    """
    Core of the fitting procedure.
    It computes the initial guess, save the figure, then starts the fit.
    After the fit, writes the output file and saves the figures of the result.
    Summary of saved files:
    > "<filename>.out": fit report
    > "<filename>_iguess.<ext>": figure of the initial guess
    > "<filename>_total.<ext>": figure that contains the experimental spectrum, the total fitting function, and the residuals
    > "<filename>_wcomp.<ext>": figure that contains the experimental spectrum, the total fitting function, and the components in different colors. The residuals are not shown
    > "<filename>_rhist.<ext>": histogram of the residual, with a gaussian function drawn on top according to its statistical parameters.
    ----------
    Parameters:
    - M: kz.Spectrum_1D object
        Mixture spectrum
    - N_spectra: int
        Number of spectra to be used as fitting components
    - Lparam: lmfit.Parameters object
        Normalized parameters
    - param: lmfit.Parameters object
        Actual parameters
    - lims: list of tuple or None
        Delimiters of the fitting region, in ppm. If None, the whole spectrum is used.
    - fit_kws: dict of keyworded arguments
        Additional parameters for the lmfit.Minimizer.minimize function
    - filename: str
        Root of the names for the names of the files that will be saved.
    - ext: str
        Format of the figures
    - dpi: int
        Resolution of the figures, in dots per inches
    """
    # Get the parameters for building the spectra
    acqus = dict(M.acqus)
    N = M.r.shape[-1]

    # Add the nucleus to the xlabel
    X_label = '$\delta\ $'+kz.misc.nuc_format(M.acqus['nuc'])+' /ppm'

    # Make a shallow copy of the experimental spectrum
    exp = np.copy(M.r)

    # Convert the limits in ppm into a slice, using the ppm scale as reference
    if lims is None:
        plims = [slice(0, -1)]
    else:
        pts = [tuple([kz.misc.ppmfind(M.ppm, lim)[0] for lim in X]) for X in lims]
        plims = [slice(min(W), max(W)) for W in pts]

    # Trim the spectrum according to the lims
    exp_T = np.concatenate([exp[w] for w in plims])

    # Calculate initial spectra
    i_spectra = calc_spectra(Lparam, param, N_spectra, acqus, N)
    # Initial guess of the total calculated spectrum
    i_total = np.sum([s for s in i_spectra], axis=0)
    i_total_T = np.concatenate([i_total[w] for w in plims])
    # Calculate an intensity correction factor
    I, _ = kz.fit.fit_int(exp_T, i_total_T)             

    # Plot the initial guess
    print('Saving figure of the initial guess...')
    plots.plot_iguess(M.ppm, exp, I*i_total, [I*s for s in i_spectra], 
            lims=(np.max(np.array(lims)), np.min(np.array(lims))), 
            X_label=X_label, filename=filename, ext=ext, dpi=dpi)
    print('Done.\n')

    param.add('count', value=0, vary=False)
    # Make a file for saving the convergence path
    cnvg_path = f'{filename.rsplit(".")[0]}.cnvg'
    # Clear it and write the header
    with open(cnvg_path, 'w') as cnvg:
        cnvg.write('# Step \t Target\n')

    # Do the fit
    @kz.cron
    def start_fit():
        print(f'This fit has {len(Lparam)} parameters.\nStarting fit...')
        minner = l.Minimizer(f2min, Lparam, fcn_args=(param, N_spectra, acqus, N, exp_T, I, plims, cnvg_path))
        result = minner.minimize(**fit_kws)
        print(f'{result.message}\nNumber of function evaluations: {result.nfev}.')
        return result
    result = start_fit()

    # Get the optimized parameters
    Lpopt = result.params
    # Calculate the optimized spectra
    #   ...as arrays
    opt_spectra = calc_spectra(Lpopt, param, N_spectra, acqus, N)
    #   ...as kz.fit.Peak objects
    opt_spectra_obj = calc_spectra_obj(Lpopt, param, N_spectra, acqus, N)

    # Normalize the intensities so that they sum up to 1
    #   Get the actual intensities
    K = [f for key, f in param.valuesdict().items() if 'I' in key]
    #   Normalize them
    K_norm, I_corr = kz.misc.molfrac(K)
    #   Correct the total intensity to preserve the absolute values
    I_abs = I * I_corr
    
    opt_total = np.sum(opt_spectra, axis=0)
    # Write the output
    write_output(M, I_abs, K_norm, opt_spectra_obj, 
            lims=(np.max(np.array(lims)), np.min(np.array(lims))), 
            filename=f'{filename}.out')
    print(f'The results of the fit are saved in {filename}.out.\n')
    
    # Make the plot of the convergence path
    plots.convergence_path(cnvg_path, filename=f'{filename}_cnvg', ext=ext, dpi=dpi)

    # Make the figures
    print('Saving figures...')
    plots.plot_output(M.ppm, exp, I*opt_total, [I*s for s in opt_spectra], 
            lims=(np.max(np.array(lims)), np.min(np.array(lims))), 
            plims=plims,
            X_label=X_label, filename=filename, ext=ext, dpi=dpi)
    print('Done.\n')


        

        



