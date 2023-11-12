#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import klassez as kz
import lmfit as l

def as_par(name, value, lims=0, rel=True):
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
            else:
                minval = value - lims
                maxval = value + lims

        # Now create the Parameter object with the given values
        p = l.Parameter(
                name = f'{name}',
                value = value,
                min = minval,
                max = maxval,
                )
    return p

def P2L(P):
    """
    Normalize a lmfit.Parameter object according to its boundaries. Works only if expr is not set! In this case, in fact, it returns None.
    The boundaries of the new parameter are set to be (0,1), where 0 corresponds to P.min and 1 to P.max.
    ------
    Parameters:
    - P: lmfit.Parameter object
        Not normalized parameter
    ------
    Returns:
    - L: lmfit.Parameter object
        Normalized parameter. If P.expr is set, this is None.
    """
    if P.expr is None or P.expr == '': 
        L = l.Parameter(
                name = f'L{P.name}',
                value = (P.value - P.min) / (P.max - P.min),
                min = 0,
                max = 1,
                )
        return L
    else:   # Do nothing and return None
        return

def L2P(L, Xmin, Xmax):
    """
    Convert a normalized parameter into its absolute counterpart.
    ------
    Parameters:
    - L: float
        Normalized parameter value
    - Xmin: float
        Lower bound of the "original" patameter
    - Xmax: float
        Upper bound of the "original" patameter
    ------
    Returns:
    - name: str
        Label of the parameter
    - value: float
        Correspondant value
    """
    value = L * (Xmax - Xmin) + Xmin
    return value


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
        rel=False
        ))
    #   intensity
    p.append(as_par(
        f'S{spect}_k{idx}',
        dic['k'],
        bds['ktol'],
        rel=False
        ))
    #   x_g
    p.append(as_par(
        f'S{spect}_x_g{idx}',
        dic['x_g'],
        (0, 1),
        rel=False
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
            ))
        # intensity
        p.append(as_par(
            f'S{spect}_k{idx}',
            dic['k'],
            bds['ktol'],
            rel=False,
            ))
        #   x_g
        p.append(as_par(
            f'S{spect}_x_g{idx}',
            dic['x_g'],
            (0, 1),
            rel=False
            ))
    return p


def main(M, components, bds):
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
    -----------
    Returns:
    - Lparam: lmfit.Parameters object
        Normalized parameters for the fit
    - param: lmfit.Parameters object
        Actual parameters for the fit
    """

    # Get acqus and the spectra as collection of peaks
    acqus = dict(M.acqus)
    N = M.r.shape[-1]
    N_spectra = len(components) # Number of spectra

    # Create the parameter object
    param = l.Parameters()
    for k, S in enumerate(components):  # Loop on the spectra
        # Intensity
        param.add(as_par(f'S{k+1}_I', 1/N_spectra, (0, 10)))
        # All the other parameters
        for group, multiplet in S.p_collections.items():
            if group == 0:  # Group 0 is a list!
                for peak in multiplet:
                    # Make the parameters
                    p = singlet2par(peak, f'{k+1}', bds)
                    for par in p:
                        # Add them by unpacking the list
                        param.add(par)
            else:   
                # make the parameters
                p = multiplet2par(multiplet, f'{k+1}', group, bds)
                # Add them by unpacking the list
                for par in p:
                    param.add(par)
    # Normalize the parameters
    Lparam = l.Parameters() # Normalized parameters
    for name in param:
        Q = P2L(param[name])    # Make the conversion
        if Q is not None:   # Add only the ones which has not an expression set, to avoid errors
            Lparam.add(Q)

    return Lparam, param
