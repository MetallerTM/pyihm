#! /usr/bin/env python3

# header
import os
from datetime import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt
import klassez as kz
from gen_param import main as gen_param
from gen_param import L2P

import lmfit as l


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
            x_g=0, 
            phi=0,
            N=N,
            ) for i in peaks_idx[n]]
        # Add the list of peaks to the final list
        spectra.append(peak_list)
    return spectra

def f2min(Lparam, param, N_spectra, acqus, N, exp, I):
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
    ----------
    Returns:
    - target: float
        \sum [ (exp - I*calc)^2 ]
    """
    # Compute the trace for each spectrum
    spectra = calc_spectra(Lparam, param, N_spectra, acqus, N)

    # Sum the spectra to give the total fitting trace
    total = np.sum(spectra, axis=0)
    # Make the residuals
    residual = exp - I * total
    target = np.sum(residual**2)
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
        

def main(M, N_spectra, Lparam, param):
    import plots
    # Get the parameters for building the spectra
    acqus = dict(M.acqus)
    N = M.r.shape[-1]

    # Make a shallow copy of the experimental spectrum
    exp = np.copy(M.r)

    # Calculate initial spectra
    i_spectra = calc_spectra(Lparam, param, N_spectra, acqus, N)
    # Initial guess of the total calculated spectrum
    i_total = np.sum([s for s in i_spectra], axis=0)
    # Calculate an intensity correction factor
    I, _ = kz.fit.fit_int(exp, i_total)

    # Plot the initial guess
    #plots.plot_iguess(M.ppm, exp, I*i_total, [I*s for s in i_spectra], lims=(10,0))

    # Do the fit
    print('Starting fit...')
    minner = l.Minimizer(f2min, Lparam, fcn_args=(param, N_spectra, acqus, N, exp, I))
    result = minner.minimize(method='nelder', max_nfev=10000, tol=1e-15)
    print(result.message, result.nfev)

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
    
    # Write the output
    write_output(M, I_abs, K_norm, opt_spectra_obj, (max(M.ppm), min(M.ppm)), filename='fit.report')

    plt.figure()
    plt.plot(exp, label='E')
    plt.plot(I * np.sum(opt_spectra, axis=0), lw=0.6, label='F', c='r')
    plt.legend()

    plt.figure()
    plt.plot(exp, label='E')
    for k, s in enumerate(opt_spectra):
        plt.plot(I * s, lw=0.5, label=f'{k+1}')
    plt.legend()

    plt.figure()
    plt.plot(exp - I * np.sum(opt_spectra, axis=0), 'b.', label='R')
    plt.legend()
    plt.show()


        

        



