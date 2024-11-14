#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import klassez as kz
import lmfit as l

def as_par(name, value, lims=0, rel=True, minthresh=None):
    """
    Creates a lmfit.Parameter object using the given parameters.
    ---------
    Parameters:
    - name: str
        Label of the parameter
    - value: float or str
        If it is float, it is the value of the parameter. If it is a str, it is put in the 'expr' attribute of the lmfit.Parameter object.
    - lims: float or tuple
        Determines the boundaries. If it is a tuple, the boundaries are min(lims) and max(lims). If it is a single float, the boundaries are (value-lims, value+lims). Not read if value is str
    - rel: bool
        Relative boundaries. If it is True and lims is a float, the boundaries are set to value-lims*value, value+lims*value.
    - minthresh: float
        If given, overwrite the minimum threshold with this value, if the calculated one is lower than it.
    ---------
    Returns:
    - p: lmfit.Parameter object
        Object created according to the given parameter
    """
    # Check if value is a string or a float
    if isinstance(value, str):  # It is expr
        p = l.Parameter(
                name = f'{name}',
                expr = value,
                )
    else:   # We have to set also the boundaries
        # Check if lims is a sequence
        if isinstance(lims, (tuple, list, np.ndarray)):
            # Set the minimum and maximum values accordingly
            minval, maxval = min(lims), max(lims)
        else:
            # Discriminate between relative or absolute limits
            if rel is True:
                minval = value - lims*value
                maxval = value + lims*value
                if minval == maxval:
                    minval = value - 1e-10
                    maxval = value + 1e-10
            else:
                minval = value - lims
                maxval = value + lims


        # Replace lower boundary with absolute value
        if minthresh is not None:
            minval = max(minval, minthresh)

        # Now create the Parameter object with the given values
        p = l.Parameter(
                name = f'{name}',
                value = value,
                min = minval,
                max = maxval,
                )
    return p

def singlet2par(item, spect, bds):
    """
    Converts a fit.Peak object into a list of lmfit.Parameter objects: the chemical shift (u), the linewidth (s), and intensity (k).
    The keys are of the form 'S#_p?' where # is spect and ? is the index of the peak.
    --------
    Parameters:
    - item: kz.fit.Peak object
        Peak to convert into Parameter. Make sure the .idx attribute is set!
    - spect: int
        Label of the spectrum to which the peak belongs to
    - bds: dict
        Contains the parameters' boundaries
    -------
    Returns:
    - p: list
        List of lmfit.Parameter objects
    """
    # Get index of the peak
    idx = item.idx
    # Get the parameters of the peak
    dic = item.par()
    # Create empty list
    p = []
    ## Create the Parameter objects
    #   chemical shift
    p.append(as_par(
        f'S{spect}_u{idx}',
        dic['u'],
        bds['utol'],
        rel = False,
        ))
    #   linewidth
    p.append(as_par(
        f'S{spect}_s{idx}',
        dic['fwhm'],
        bds['stol'],
        rel=False,
        minthresh=0,
        ))
    #   intensity
    p.append(as_par(
        f'S{spect}_k{idx}',
        dic['k'],
        bds['ktol'],
        rel=False,
        minthresh=0,
        ))
    #   b
    p.append(as_par(
        f'S{spect}_b{idx}',
        dic['b'],
        (0, 1),
        rel=False,
        ))
    return p

def multiplet2par(item, spect, group, bds):
    """
    Converts a Multiplet object into a list of lmfit.Parameter objects.
    The keys are of the form 'S#_p?' where # is spect and ? is the index of the peak.
    p = U is the mean chemical shift
    p = o is the offset from U
    p = u is the absolute chemical shift, computed as U + o, set as expression.
    --------
    Parameters:
    - item: fit.Peak object
        Peak to convert into Parameter. Make sure the .idx attribute is set!
    - spect: int
        Label of the spectrum to which the peak belongs to
    - group: int
        Label of the multiplet group
    - bds: dict
        Contains the parameters' boundaries
    -------
    Returns:
    - p: list
        List of lmfit.Parameter objects
    """
    p = []
    for idx, dic in item.par().items():
        # chemical shift, total
        p.append(as_par(
            f'S{spect}_U{group}',
            dic['U'],
            bds['utol'],
            rel = False,
            ))
        # chemical shift, offset from U
        p.append(as_par(
            f'S{spect}_o{idx}',
            dic['u_off'],
            bds['utol_sg'],
            rel = False,
            ))
        p.append(as_par(
            f'S{spect}_u{idx}',
            f'S{spect}_U{group} + S{spect}_o{idx}',
            0.01,   # Meaningless, just placeholder
            rel = False,
            ))
        # linewidth
        p.append(as_par(
            f'S{spect}_s{idx}',
            dic['fwhm'],
            bds['stol'],
            rel=False, 
            minthresh=0,
            ))
        # intensity
        p.append(as_par(
            f'S{spect}_k{idx}',
            dic['k'],
            bds['ktol'],
            rel=False,
            minthresh=0,
            ))
        #   b
        p.append(as_par(
            f'S{spect}_b{idx}',
            dic['b'],
            (0, 1),
            rel=False
            ))
    return p


def main(M, components, bds, lims, Hs, c_idx, I0=None):
    """
    Create the lmfit.Parameters objects needed for the fitting procedure.
    -----------
    Parameters:
    - M: kz.Spectrum_1D object
        Mixture spectrum
    - components: list
        List of Spectra objects
    - bds: dict
        Boundaries for the fitting parameters.
    - lims: list of tuple
        Borders of the fitting windows, in ppm (left, right)
    - Hs: list
        Number of protons each spectrum integrates for
    - c_idx: list
        index of the components that are actually used in the fit
    - I0: list 
        Initial guess for the concentrations. If None, 1 is used for each spectrum
    -----------
    Returns:
    - param: lmfit.Parameters object
        Actual parameters for the fit
    """
    # Get acqus and the spectra as collection of peaks
    acqus = dict(M.acqus)
    N = M.r.shape[-1]
    N_spectra = len([w for w in components if w != 'Q']) # Number of spectra
    if I0 is None:
        I0 = [1. for k in range(N_spectra)]

    # Create the parameter object
    param = l.Parameters()
    for k, S in enumerate(components):  # Loop on the spectra
        # Intensity
        param.add(as_par(f'S{c_idx[k]+1}_I', I0[k], (0, np.sum(Hs))))
        # All the other parameters
        for group, multiplet in S.p_collections.items():
            if group == 0:  # Group 0 is a list!
                for peak in multiplet:
                    # Make the parameters
                    p = singlet2par(peak, f'{c_idx[k]+1}', bds)
                    for par in p:
                        # Add them by unpacking the list
                        param.add(par)
            else:   
                # make the parameters
                p = multiplet2par(multiplet, f'{c_idx[k]+1}', group, bds)
                # Add them by unpacking the list
                for par in p:
                    param.add(par)

    # Correct the boundaries of the chemical shifts to make them stay in the fitting windows
    for p in param:
        # Search the chemical shifts
        if 'u' in p or 'U' in p:
            # Look for the region where the u falls in
            for lim in lims:
                if min(lim) <= param[p].value and param[p].value <= max(lim):
                    limits = lim
                    break
            # Update the limits:
            # the left border is the right-most value between the actual left border and the left border of the fitting window
            # the right border is the left-most value between the actual right border and the right border of the fitting window
            param[p].set(min=max(min(lim), param[p].min), max=min(max(lim), param[p].max) ) 

    return param
