#! /usr/bin/env python3

import os
from datetime import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt
import klassez as kz
import lmfit as l

from .gen_param import main as gen_param
from . import plots
from . import GUIs


def calc_spectra(param, N_spectra, acqus, N):
    """
    Computes the spectra to be used as components for the fitting procedure, in form of lists of 1darrays. Each array is the sum of all the peaks.
    This function is called at each iteration of the fit.
    ---------
    Parameters:
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
    # Separate the parameters according to the spectra
    d_param = param.valuesdict()    # Convert to dictionary
    keys = list(d_param.keys())     # Get the keys
    spectra_par = []    # Placeholder

    component_idx = [int(key.split('_')[0].replace('S', '')) for key in param if 'I' in key]
    for n in component_idx:  # read: for each spectrum
        # Make a list of dictionaries. Each dictionary contains only the parameters relative to a given spectrum
        spectra_par.append({key.replace(f'S{n}_', ''): d_param[key] for key in keys if f'S{n}_' in key})


    # How many peaks there are in each spectrum?
    peaks_idx = []  # Placeholder
    for par in spectra_par: # read: for each spectrum
        # Get the indices of the peaks 
        idxs = [eval(key.replace(f'u', '')) for key in par if 'u' in key]
        # Append it in the list
        peaks_idx.append(idxs)

    # Now we make the spectra!
    spectra = []    # Placeholder
    for n in range(N_spectra): # read: for each spectrum
        dic = dict(spectra_par[n])  # Alias for the n-th spectrum peaks
        # Generate the fit.Peak objects
        peak_list = [kz.fit.Peak(acqus,
            u=dic[f'u{i}'], 
            fwhm=dic[f's{i}'],
            k=dic[f'k{i}'],
            b=dic[f'b{i}'], 
            phi=0,
            N=N,
            ) for i in peaks_idx[n]]
        # Compute the trace for each peak, then sum them up, finally multiply by the intensity
        spectra.append(dic['I'] * np.sum([peak() for peak in peak_list], axis=0))
    return spectra

def calc_spectra_obj(param, N_spectra, acqus, N):
    """
    Computes the spectra to be used as components for the fitting procedure, in form of lists of kz.fit.Peak objects. 
    ---------
    Parameters:
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
    # Separate the parameters according to the spectra
    d_param = param.valuesdict()    # Convert to dictionary
    keys = list(d_param.keys())     # Get the keys
    spectra_par = []    # Placeholder

    component_idx = [int(key.split('_')[0].replace('S', '')) for key in param if 'I' in key]
    for n in component_idx:  # read: for each spectrum
        # Make a list of dictionaries. Each dictionary contains only the parameters relative to a given spectrum
        spectra_par.append({key.replace(f'S{n}_', ''): d_param[key] for key in keys if f'S{n}_' in key})

    # How many peaks there are in each spectrum?
    peaks_idx = []  # Placeholder
    for par in spectra_par: # read: for each spectrum
        # Get the indices of the peaks 
        idxs = [eval(key.replace(f'u', '')) for key in par if 'u' in key]
        # Append it in the list
        peaks_idx.append(idxs)

    # Now we make the spectra!
    spectra = []    # Placeholder
    for n in range(N_spectra):  # read: for each spectrum
        dic = dict(spectra_par[n])  # Alias for the n-th spectrum peaks
        # Generate the fit.Peak objects
        peak_list = [kz.fit.Peak(acqus,
            u=dic[f'u{i}'], 
            fwhm=dic[f's{i}'],
            k=dic[f'k{i}'],
            b=dic[f'b{i}'], 
            phi=0,
            N=N,
            ) for i in peaks_idx[n]]
        # Add the list of peaks to the final list
        spectra.append(peak_list)
    return spectra

def f2min_align(param, N_spectra, acqus, N, exp, plims, debug=False):
    """
    Function to compute the quantity to be minimized by the fit.
    ----------
    Parameters:
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
    - plims: slice
        Delimiters for the fitting region. The residuals are computed only in this regio. They must be given as point indices
    - debug: bool
        True for saving a figure of the ongoing fit every 20 iterations
    ----------
    Returns:
    - t_residual: 1darray
        exp/I - calc
    """
    param['count'].value += 1
    count = param['count'].value
    # Compute the trace for each spectrum
    spectra = calc_spectra(param, N_spectra, acqus, N)
    # Sum the signals to give the total fitting trace
    total = np.sum([s for s in spectra], axis=0)
    # Cut the total traces according to the fitting windows
    total_T = [total[w] for w in plims]

    # Make the integrals for each fitting window
    F_total = [kz.processing.integral(s) for s in total_T]

    R = []      # Placeholder for residuals
    calc = [] # Placeholder for total fitting trace

    for E, C in zip(exp, F_total):      # Loop on the fitting windows
        # Calculate the intensity and offset factors
        intensity, offset = kz.fit.fit_int(E, C)    
        # Correct the calculated spectrum for these values
        calc.append(intensity * C + offset)
        # Compute the residuals
        tmp_res = E - (intensity * C + offset)
        R.append(tmp_res)
    # Make experimental and calculated spectrum a 1darray by concatenating the windows
    F_exp = np.concatenate(exp)
    F_calc = np.concatenate(calc)
    # Make the residuals a 1darray by concatenating the windows
    t_residual = np.concatenate(R)

    if debug:
        # FIGURE
        if (count-1) % 20 == 0:
            kz.figures.ongoing_fit(
                    F_exp, F_calc, t_residual,
                    filename='ongoing_align', dpi=200
                    )
            kz.figures.ongoing_fit(
                    np.concatenate([np.gradient(y) for y in exp]),
                    np.concatenate([np.gradient(y) for y in calc]),
                    np.concatenate([np.gradient(y) for y in R]),
                    filename='ongoing_align_spectrum', dpi=200
                    )

    # Compute the target value and print it
    target = np.sum(t_residual**2) / len(t_residual)
    print(f'Iteration step: {count:5.0f}; Target: {target:10.5e}', end='\r')

    return t_residual

def pre_alignment(exp, acqus, N_spectra, N, plims, param, DEBUG_FLAG=False):
    """
    Makes a fit with all the parameters blocked, except for the chemical shifts, on the target function of the integral.
    Used to improve the initial guess in case of misplacements of the signals.
    ----------
    Parameters:
    - exp: 1darray
        Experimental spectrum
    - acqus: dict
        Dictionary of acquisition parameters
    - N_spectra: int
        Number of spectra to be used as components
    - N: int
        Number of points for zero-filling, i.e. final dimension of the arrays
    - plims: list of slice
        Delimiters for the fitting region. The residuals are computed only in these regions. They must be given as point indices
    - param: lmfit.Parameters object
        actual parameters
    - DEBUG_FLAG: bool
        True for saving a figure of the ongoing fit every 20 iterations
    ----------
    Returns:
    - popt: lmfit.Parameters object
        Parameters with optimal chemical shifts
    """

    # Cut the experimental spectrum according to the fitting windows
    exp_T = [exp[w] for w in plims]
    # Compute the integrals of the experimental spectrum for each window
    Fexp_T = [kz.processing.integral(f) for f in exp_T]
    # Normalize it to make smaller numbers
    Fexp_T = [s / np.max(np.concatenate(Fexp_T)) for s in Fexp_T]

    # Add the fit counter
    param.add('count', value=0, vary=False)
    # Store the 'vary' status of all the parameters
    vary_dict = {}
    for p in param: # Loop on the parameters name
        vary_dict[p] = param[p].vary    # Store
        # Block all parameters that are not chemical shifts
        if 'u' in p or 'U' in p:
            pass
        else:
            param[p].set(vary=False)

    print(f'The alignment fit has {len([key for key in param if param[key].vary])} parameters.')
    # Make the fit
    @kz.cron
    def start_fit_align():
        print('Starting alignment fit...')
        minner = l.Minimizer(f2min_align, param, fcn_args=(N_spectra, acqus, N, Fexp_T, plims, DEBUG_FLAG))
        result = minner.minimize(method='leastsq', max_nfev=20000, xtol=1e-8, ftol=1e-8, gtol=1e-8)
        print(f'Alignment {result.message} Number of function evaluations: {result.nfev}.')
        return result
    result = start_fit_align()
    popt = result.params

    # Reset the "vary" status of the parameters to the original one
    for p in param:
        popt[p].set(vary=vary_dict[p])
        # Decrease the variation window of the chemical shifts to 1/10 of the starting one
        if 'u' in p or 'U' in p:
            # Get the variation span
            var_win = np.abs(popt[p].max - popt[p].min)
            # utol is 1/2 of var_win, hence to get 1/10 you have to divide by 20
            new_utol = var_win / 20
            # update the values
            popt[p].set(min = popt[p].value - new_utol)
            popt[p].set(max = popt[p].value + new_utol)

    return popt


def f2min(param, N_spectra, acqus, N, exp, I, plims, cnvg_path, debug=False):
    """
    Function to compute the quantity to be minimized by the fit.
    ----------
    Parameters:
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
    - debug: bool
        If True, saves a figurte of the ongoing fit in the current working directory every 20 iterations
    ----------
    Returns:
    - t_residual: 1darray
        exp/I - calc
    """
    param['count'].value += 1
    count = param['count'].value
    # Compute the trace for each spectrum
    spectra = calc_spectra(param, N_spectra, acqus, N)
    spectra_T = [np.concatenate([spectrum[w] for w in plims]) for spectrum in spectra]

    # Sum the spectra to give the total fitting trace
    total = np.sum(spectra_T, axis=0)

    t_residual = exp / I - total

    target = np.sum(t_residual**2) / len(t_residual)

    if debug:
        if (count-1) % 20 == 0:
            kz.figures.ongoing_fit(exp/I, total, t_residual, filename='ongoing_fit', dpi=200)
            if 1:
                npts = 0
                for k, w in enumerate(plims):
                    lenw = w.indices(N)[1] - w.indices(N)[0]
                    kz.figures.ongoing_fit(exp[npts:npts+lenw]/I, total[npts:npts+lenw], t_residual[npts:npts+lenw], filename='ongoing_fit_W'+f'{k+1}', dpi=200)
                    npts += lenw

    # Print how the fit is going, both in the file and in standart output
    with open(cnvg_path, 'a', buffering=1) as cnvg:
        cnvg.write(f'{count:5.0f}\t{target:10.5e}\n')
    print(f'Iteration step: {count:5.0f}; Target: {target:10.5e}', end='\r')

    return t_residual


def write_output(M, I, K, spectra, n_comp, lims, filename='fit.report'):
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
        Relative concentrations of the components spectra in the mixture
    - spectra: list of kz.fit.Peak objects
        Computed components of the mixture, weighted for their relative intensity
    - n_comp: list
        Indices of the components of the mixture
    - lims: tuple
        Upper and lower boundaries of the fit region
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
    f.write('! Fit performed by {} on {}\n\n'.format(os.getlogin(), date_and_time))
    f.write(f'Mixture spectrum: {os.path.join(M.datadir, M.filename)}\n\n')
    f.write(f'Absolute intensity correction: I = {I:.8e}\n\n')
    f.write('Relative intensities:\n')
    for k, r_i in enumerate(K):
        f.write(f'Component {n_comp[k]:>3.0f}: {r_i*100:10.5f}% | Rel: {r_i/min(K):10.5f}\n')
    f.write('\n\n\n')
    f.close()

    ## PARAMETERS OF THE PEAKS
    for k, component in enumerate(spectra):
        # Make a dictionary of peak objects in order to use kz.fit.write_vf
        dict_component = {j+1: peak for j, peak in enumerate(component)}
        # Spacer for the spectrum identifier
        with open(filename, 'a', buffering=1) as f:
            f.write(f'Component {n_comp[k]} fitted parameters:\n')
        # Do the writing
        kz.fit.write_vf(filename, dict_component, lims, K[k]*I)
        # Add space
        with open(filename, 'a', buffering=1) as f:
            f.write(f'\n\n')

def save_data(filename, ppm_scale, exp, *opt_spectra):
    """
    Saves the ppm scale, the experimental spectrum, the total trace and the components in .csv files, to be opened with excel, origin, or whatever.
    ---------
    Parameters:
    - filename: str
        Location of the filename to be saved, without the .csv extension.
    - ppm_scale: 1darray
        PPM scale of the experimental spectrum
    - exp: 1darray
        Experimental spectrum, real part
    - opt_spectra: sequence of 1darray
        Spectra of the components
    """
    # Make the header for the columns
    header = ['PPM_SCALE', 'EXPERIMENTAL', 'TOTAL'] + [f'COMPONENT_{k+1}' for k in range(len(opt_spectra))]
    # Assemble all the information in a single 2darray
    data = [ ppm_scale, exp, np.sum(opt_spectra, axis=0)] + [ y for y in opt_spectra]
    # Transpose to make it column_wise
    data_arr = np.array(data).T
    # Save the file. "comments=''" is needed because header appears as a comment, therefore adds a column
    np.savetxt(f'{filename}.csv', data_arr, header=' '.join(header), comments='')

        

def main(M, N_spectra, Hs, param, I, lims=None, fit_kws={}, filename='fit', NOALGN_FLAG=False, DEBUG_FLAG=False, METHOD_FLAG='fast', ext='tiff', dpi=600):
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
    - Hs: list
        Number of protons each spectrum integrates for
    - param: lmfit.Parameters object
        Actual parameters
    - lims: list of tuple or None
        Delimiters of the fitting region, in ppm. If None, the whole spectrum is used.
    - fit_kws: dict of keyworded arguments
        Additional parameters for the lmfit.Minimizer.minimize function
    - filename: str
        Root of the names for the names of the files that will be saved.
    - NOALGN_FLAG: bool
        If True, skips the alignment fit
    - DEBUG_FLAG: bool
        True for saving a figure of the ongoing fit every 20 iterations
    - METHOD_FLAG: str
        Method to be used for the fit. Can be 'fast', 'tight', 'custom'
    - ext: str
        Format of the figures
    - dpi: int
        Resolution of the figures, in dots per inches
    """
    # Get the parameters for building the spectra
    acqus = dict(M.acqus)
    acqus['freq'] = M.freq
    N = M.r.shape[-1]

    # Get names for the figures
    if os.sep in filename:
        base_dir, name = filename.rsplit(os.sep, 1)
    else:
        base_dir = os.getcwd()
        name = filename

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
    i_spectra = calc_spectra(param, N_spectra, acqus, N)
    i_spectra_obj = calc_spectra_obj(param, N_spectra, acqus, N)

    # Initial guess of the total calculated spectrum
    i_total = np.sum([s for s in i_spectra], axis=0)
    i_total_T = np.concatenate([i_total[w] for w in plims])
    # Calculate an intensity correction factor

    # Plot the initial guess
    print('Saving figure of the initial guess...')
    plots.plot_iguess(M.ppm, exp/I, i_total, [s for s in i_spectra], 
            lims=(np.max(np.array(lims)), np.min(np.array(lims))), plims=plims, 
            X_label=X_label, filename=os.path.join(base_dir, f'{name}-FIGURES', filename), ext=ext, dpi=dpi)
    # Save the data in a .csv file
    save_data(os.path.join(base_dir, f'{name}-DATA', f'{filename}-iguess'), M.ppm, M.r, *[I*y for y in i_spectra])
    print('Done.\n')

    param.add('count', value=0, vary=False)

    if not NOALGN_FLAG:
        # Compute residuals before the alignment
        R_before_algn = f2min(param, N_spectra, acqus, N, exp_T, I, plims, '/dev/null', debug=False)
        param['count'].set(value=0)
        target_before_algn = np.sum(R_before_algn**2) / len(R_before_algn)   
        
        # Align the chemical shifts
        param = pre_alignment(exp, acqus, N_spectra, N, plims, param, DEBUG_FLAG)

        # Compute residuals before the alignment
        R_after_algn = f2min(param, N_spectra, acqus, N, exp_T, I, plims, '/dev/null', debug=False)
        target_after_algn = np.sum(R_after_algn**2) / len(R_after_algn)
        param['count'].set(value=0)

        # Check if the residuals are worse
        if target_after_algn > target_before_algn:
            print('WARNING: The alignment fit seems to have failed.')
            print(f'Target went from {target_before_algn} to {target_after_algn}.')
            print('You might consider the option to re-run the fit with the --noalgn option.')

    # Figure of aligned iguess
    #   ...as arrays
    algn_spectra = calc_spectra(param, N_spectra, acqus, N)
    #   ...as kz.fit.Peak objects
    algn_spectra_obj = calc_spectra_obj(param, N_spectra, acqus, N)
    #   ... and finally make the total trace
    algn_total = np.sum(algn_spectra, axis=0)

    if not NOALGN_FLAG:
        print('Saving figures...')
        plots.plot_output(M.ppm, exp/I, algn_total, [s for s in algn_spectra], 
                lims=(np.max(np.array(lims)), np.min(np.array(lims))), 
                plims=plims,
                X_label=X_label, filename=os.path.join(base_dir, f'{name}-FIGURES', f'{filename}-algn'), ext=ext, dpi=dpi)
        save_data(os.path.join(base_dir, f'{name}-DATA', f'{filename}-algn'), M.ppm, M.r, *[I*y for y in algn_spectra])
        print('Done.\n')

    # Make a file for saving the convergence path
    cnvg_path = os.path.join(base_dir, f'{name}-DATA', f'{filename.rsplit(".")[0]}.cnvg')

    # Clear it and write the header
    with open(cnvg_path, 'w') as cnvg:
        cnvg.write('# Step \t Target\n')

    # Do the fit
    @kz.cron
    def start_fit(flag):
        print(f'This fit has {len([key for key in param if param[key].vary])} parameters.')
        if flag == 'fast':
            print('Default fitting method: fast')
            minner = l.Minimizer(f2min, param, fcn_args=(N_spectra, acqus, N, exp_T, I, plims, cnvg_path, DEBUG_FLAG))
            print(f'Fitting n. 1 of 1, method: leastsq\nStarting fit...')
            result = minner.minimize(method='leastsq', max_nfev=15000, xtol=1e-8, ftol=1e-8, gtol=1e-8)
            print(f'\n{result.message} Number of function evaluations: {result.nfev}.')
        elif flag == 'tight':
            print('Default fitting method: tight')
            minner = l.Minimizer(f2min, param, fcn_args=(N_spectra, acqus, N, exp_T, I, plims, cnvg_path, DEBUG_FLAG))
            print(f'Fitting n. 1 of 2, method: Nelder\nStarting fit...')
            result = minner.minimize(method='Nelder', max_nfev=10000)
            print(f'\n{result.message} Number of function evaluations: {result.nfev}.')
            params = result.params
            print(f'Fitting n. 2 of 2, method: leastsq\nStarting fit...')
            result = minner.minimize(method='leastsq', params=params, max_nfev=10000, xtol=1e-8, ftol=1e-8, gtol=1e-8)
            print(f'\n{result.message} Number of function evaluations: {result.nfev}.')
        elif flag == 'custom':
            for idx in range(len(fit_kws.keys())):  # for each fitting round...
                if idx == 0:
                    minner = l.Minimizer(f2min, param, fcn_args=(N_spectra, acqus, N, exp_T, I, plims, cnvg_path, DEBUG_FLAG))
                if fit_kws[idx]['method'] == 'leastsq':
                    tol = fit_kws[idx].pop('tol')
                    fit_kws[idx]['xtol'] = tol
                    fit_kws[idx]['ftol'] = tol
                    fit_kws[idx]['gtol'] = tol
                print(f'Fitting n. {idx+1} of {len(fit_kws.keys())}, method: {fit_kws[idx]["method"]}\nStarting fit...')
                if idx != 0:
                    fit_kws[idx]['params'] = result.params
                result = minner.minimize(**fit_kws[idx])
                print(f'\n{result.message} Number of function evaluations: {result.nfev}.')
        return result
    result = start_fit(METHOD_FLAG)

    # Get the optimized parameters
    popt = result.params
    # Calculate the optimized spectra
    #   ...as arrays
    opt_spectra = calc_spectra(popt, N_spectra, acqus, N)
    #   ...as kz.fit.Peak objects
    opt_spectra_obj = calc_spectra_obj(popt, N_spectra, acqus, N)
    Hf = np.empty(len(Hs))
    for k, peaks in enumerate(opt_spectra_obj):
        Hf[k] = np.sum([peak.k for peak in peaks])
    #   ... and finally make the total trace
    opt_total = np.sum(opt_spectra, axis=0)


    component_idx = [int(key.split('_')[0].replace('S', '')) for key in popt if 'I' in key]

    #   Get the actual intensities
    concentrations = np.array([f for key, f in popt.valuesdict().items() if 'I' in key])
    # Correction factor for intensities
    KH = Hf / Hs
    concentrations *= KH
    for k, peaks in enumerate(opt_spectra_obj):
        for peak in peaks:
            peak.k /= KH[k]

    #   Normalize them
    c_norm, I_corr = kz.misc.molfrac(concentrations)
    #   Correct the total intensity to preserve the absolute values
    I_abs = I * I_corr

    # Print relative concentrations of the components
    print('FIT RESULTS')
    for k, K in enumerate(c_norm):
        print(f'Component {component_idx[k]:2.0f}: {K*100:10.5f}% | Rel: {K/min(c_norm):10.5f}')
    print()
    
    # Write the output
    write_output(M, I_abs, c_norm, opt_spectra_obj, 
            n_comp = component_idx,
            lims=(np.max(np.array(lims)), np.min(np.array(lims))), 
            filename=f'{filename}.out')
    print(f'The results of the fit are saved in {filename}.out.\n')
    
    # Make the plot of the convergence path
    try:
        plots.convergence_path(cnvg_path, filename=os.path.join(base_dir, f'{name}-FIGURES', f'{filename}_cnvg'), ext=ext, dpi=dpi)
    except:
        pass

    # Make the figures
    print('Saving figures...')
    plots.plot_output(M.ppm, exp, I*opt_total, [I*s for s in opt_spectra], 
            lims=(np.max(np.array(lims)), np.min(np.array(lims))), 
            plims=plims,
            X_label=X_label, filename=os.path.join(base_dir, f'{name}-FIGURES', filename), ext=ext, dpi=dpi, windows=True)
    save_data(os.path.join(base_dir, f'{name}-DATA', f'{filename}-result'), M.ppm, M.r, *[I*y for y in opt_spectra])
    print('Done.\n')



