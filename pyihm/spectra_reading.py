#! /usr/bin/env python3

import os
import numpy as np
import klassez as kz
from copy import deepcopy

from . import GUIs

class Multiplet:
    """
    Class that represent a multiplet as a collection of peaks.
    --------------
    Attributes:
    - acqus: dict
        Dictionary of acquisition parameters
    - peaks: dict
        Dictionary of kz.fit.Peak objects
    - U: float
        Mean chemical shift of the multiplet
    - u_off: dict
        Chemical shift of the components of the multiplet, expressed as offset from self.U
    """
    def __init__(self, acqus, *peaks):
        """
        Initialize the class.
        ---------
        Parameters:
        - acqus: dict
            Dictionary of acquisition parameters
        - peaks: kz.fit.Peak objects
            Peaks that are part of the multiplet. They must have an attribute 'idx' which serves as label
        """
        # Store the acqus dictionary
        self.acqus = acqus
        # Store the peaks in a dictionary using their own idx attribute as key
        self.peaks = {}
        for peak in peaks:
            self.peaks[peak.idx] = peak
            self.N = peak.N
            if self.N is None:
                self.N = int(self.acqus['TD'])

        # Compute the mean chemical shift and the offsets
        self.U = np.mean([p.u for _, p in self.peaks.items()])
        self.u_off = {key: p.u - self.U for key, p in self.peaks.items()}

    def par(self):
        """
        Computes a summary dictioanary of all the parameters of the multiplet.
        ---------
        Returns:
        - dic: dict of dict
            The keys of the inner dictionary are the parameters of each single peak, the outer keys are the labels of the single components
        """
        dic = {}        # Placeholder
        for key, peak in self.peaks.items():
            # Create a dictionary for each component
            dic[key] = {
                    'U': self.U,        # This is the same for all the components
                    'u_off': self.u_off[key],   # This is the distinguish trait
                    'fwhm': peak.fwhm,  
                    'k': peak.k,
                    'b': peak.b,
                    'phi': peak.phi,
                    'group': peak.group
                    }
        return dic

    def __call__(self):
        """
        Compute the trace correspondant to the multiplet.
        --------
        Returns:
        - trace: 1darray
            Sum of the components
        """
        trace = np.zeros(self.N)  # Placeholder
        for key, peak in self.peaks.items():
            # Recompute the absolute chemical shift
            self.peaks[key].u = self.U + self.u_off[key]
            # Sum the component to the total trace
            trace += peak()
        return trace


class Spectr:
    """ 
    Class that represents a spectrum as a collection of peaks and multiplets.
    ---------
    Attributes:
    - acqus: dict
        Acquisition parameters
    - peaks: dict
        Dictionary of peaks object, labelled according to the "idx" attribute of each single peak
    - unique_groups: list
        Identifier labels for the multiplets, without duplicates
    - p_collections: dict
        Dictionary of kz.fit.Peak and Multiplet objects, labelled according to the group they belong to. In particular, self.p_collections[0] is a list of kz.fit.Peak objects, whereas all the remaining entries consist of a single Multiplet object.
    - total: 1darray
        Placeholder for the trace of the spectrum, as sum of all the peaks.
    """
    def __init__(self, acqus, *peaks):
        """
        Initialize the class.
        ---------
        Parameters:
        - acqus: dict
            Dictionary of acquisition parameters
        - peaks: kz.fit.Peak objects
            Peaks that are part of the multiplet. They must have an attribute 'idx' which serves as label
        """
        # Store the acqus dictionary
        self.acqus = acqus
        # Store the peaks in a dictionary using their own idx attribute as key
        self.peaks = {}
        for peak in peaks:
            self.peaks[peak.idx] = peak
            self.N = peak.N
            if self.N is None:
                self.N = int(self.acqus['TD'])

        ## Sort the peaks according to the 'group' attribute: this separates the multiplets
        all_groups = {key: p.group for key, p in self.peaks.items()}    # Get the group labels
        # Remove duplicates
        self.unique_groups = sorted(list(set([g for _, g in all_groups.items()])))

        self.p_collections = {} # Placeholder
        for g in self.unique_groups:    # Loop on the group labels
            # Get only the peaks of the same group
            keys = [key for key, item in all_groups.items() if item == g]
            if g == 0:  # They are independent, treated as singlets
                self.p_collections[0] = [kz.fit.Peak(self.acqus, N=self.N, **self.peaks[key].par()) for key in keys]
                # Add the labels as 'idx' attributes
                for k, key in enumerate(keys):
                    self.p_collections[0][k].idx = key 
            else:
                # Compute the multiplet which comprises the peaks of the same group
                self.p_collections[g] = Multiplet(self.acqus, *[self.peaks[key] for key in keys])
        # Compute the spectrum summing up all the collections of peaks
        self.total = self.calc_total()

    def calc_total(self):
        """
        Computes the sum of all the peaks to make the spectrum
        --------
        Returns:
        - total: 1darray
            Computed spectrum
        """
        total = np.zeros(self.N)  # Placeholder
        for g in self.unique_groups:
            if g == 0:  # Group 0 is a list of peaks!
                for s in self.p_collections[g]:
                    total += s()
            else:       # A single multiplet
                total += self.p_collections[g]()
        return total

    def __call__(self, I=1):
        """
        Compute the total spectrum, multiplied by I.
        ---------
        Parameters:
        - I: float
            Intensity value that multiplies the spectrum
        ---------
        Returns:
        - total: 1darray
            Computed spectrum
        """
        total = I * self.calc_total()
        return total


def get_component_spectrum(filename, acqus, return_dict=True, N=None, norm=True):
    """
    Reads a .ivf or .fvf file, and builds a dictionary of kz.fit.Peak objects with the information therein. Then, depending on return_dict, this function either returns it as is, or sums them up together to obtain the spectrum as array.
    ------------------
    Parameters:
    - filename: str
        Path to the .ivf or .fvf file
    - acqus: dict
        Dictionary of acquisition parameters.
    - return_dict: bool
        If True, returns the dictionary of kz.fit.Peak objects. If False, returns the computed spectrum as 1darray.
    - N: int or None
        Total number of points of the spectrum. If None, the length of the acqusition timescale is used.
    - norm: bool or float
        If True, all the peak intensities sum up to 1. If it is a number, the intensities sum up to that number. If False, they sum up to the intensity written in the .vf file
    ------------------
    Returns:
    if return_dict is True:
    - dic_Peaks: dict
        Dictionary of kz.fit.Peak parameters
    - I: float
        Total intensity 
    else:
    - total: 1darray
        Computed spectrum
    - I: float
        Total intensity 
    """
    # Compute N if not given
    if N is None:
        N = acqus['t1'].shape[-1]
    # Get the peak parameters as a list of dictionaries
    regions = kz.fit.read_vf(filename)
    I = 0
    for region in regions:
        # Remove the 'limits' entry from each region because it is useless and it would raise an error
        region.pop('limits')
        I_r = region.pop('I')
        I += I_r
        for key, peak in region.items():
            peak['k'] *= I_r
    # Join all regions together
    dic_peaks = kz.misc.merge_dict(*regions)

    # transform dic_peaks in a list of Peak objects
    dic_Peaks = {key: kz.fit.Peak(acqus, N=N, **p_par) for key, p_par in dic_peaks.items()}
    # Add the idx parameter
    for key, peak in dic_Peaks.items():
        peak.idx = key

    K = [peak.k for _, peak in dic_Peaks.items()]
    if isinstance(norm, bool):
        if norm is True:
            X, Icorr = kz.misc.molfrac(K)
            for k, key in enumerate(dic_Peaks.keys()):
                dic_Peaks[key].k = X[k]
            I *= Icorr
    elif isinstance(norm, (int, float)):
        X, Icorr = kz.misc.molfrac(K)
        for k, key in enumerate(dic_Peaks.keys()):
            dic_Peaks[key].k = X[k] * norm
        I *= Icorr / norm

    if return_dict is True:
        # return the dictionary of Peak objects
        return dic_Peaks, I
    else:
        # Compute the spectrum and return the array
        total = np.sum([peak() for _, peak in dic_Peaks.items()], axis=0)
        return total, I


def main(M, spectra_dir, Hs, lims=None, I0=None, cal_flag=False, rav_flag=False):
    """
    Reads the .fvf files, containing the fitted parameters of the peaks of a series of spectra.
    Then, computes a list of Spectr objects with those parameters, and returns it.
    The relative intensities are referred to the total intensity of the whole spectrum, not to the ones of the fitted regions.
    Employs kz.fit.read_vf to read the .fvf files and generate the parameters.
    ----------
    Parameters:
    - M: kz.Spectrum_1D object
        Mixture spectrum. Used to get the spectral parameters for the kz.fit.Peak objects
    - spectra_dir: list of str
        Sequence of the locations of the .fvf files to be read
    - Hs: list
        Number of protons each spectrum integrates for
    - lims: list of tuple
        Borders of the fitting windows, in ppm (left, right)
    - I0: list
        Initial guess for the concentrations
    - cal_flag: bool
        If True, opens the optimization of the initial guess through GUI
    - rav_flag: bool
        If True, the GUIs will be drawn in colorblind palette
    ----------
    Returns:
    - components: list of Spectr
        Spectra to be used as model in the fit. Contain only the peaks within the fittin regions.
    - Hs: list
        Corrected integrals to be used for quantification
    - I0: list
        Initial concentrations of the components in the mixture
    - lims: list of tuple
        Fitting windows, in ppm
    - c_idx: list
        Indices of the components to be used
    - I: float
        Theoretical integral of the mixture spectrum, normalized
    """
    def is_in(x, Bs):
        """ Check if the chemical shift is inside one of the fitting intervals """
        flag = False    # Default
        for B in Bs:    # Loop on the intervals
            if min(B) <= x and x <= max(B): # Check if it is inside
                flag = True     # Match as found!
                break           # Exit the loop
            else:
                pass
        return flag

    # Get "structural" parameters from M
    acqus = dict(M.acqus)
    N = M.r.shape[-1]       # Number of points for zero-filling
    _spectra_dir = deepcopy(spectra_dir)

    cal_file_flag = None
    for k, filename in enumerate(_spectra_dir):
        # <filename>.fvf becomes <filename>-cal.fvf
        base_name, extension = filename.rsplit('.', 1)
        new_filename = base_name + '-cal.' + extension

        if os.path.isfile(new_filename):
            if cal_file_flag is None:
                print(f'{base_name}-cal.{extension} file found. Do you want to load it instead?')
                load_cal = input('["n"]: no | "y": yes | "N": no all | "Y": yes all > ')
                if load_cal.isspace() or 'n' in load_cal:
                    continue
                elif 'y' in load_cal:
                    pass
                elif 'Y' in load_cal:
                    cal_file_flag = True
                elif 'N' in load_cal:
                    break
                else:
                    continue
            spectra_dir[k] = new_filename

    # List of dictionary of kz.fit.Peak objects
    comp_peaks = [get_component_spectrum(file, acqus, True, norm=Hs[k], N=M.r.shape[-1])[0] for k, file in enumerate(spectra_dir)]
    # List of 1darrays (one per spectrum)
    components = [get_component_spectrum(file, acqus, False, norm=Hs[k], N=M.r.shape[-1])[0] for k, file in enumerate(spectra_dir)]

    exit_code = 1    # Placeholder
    # Compute intensity
    I = kz.processing.integrate(M.r, x=M.freq) / (acqus['SW']/2) /np.sum(Hs)
    # Compute initial guess for concentrations
    if I0:
        Icorr = [C for C in I0]
    else:
        Icorr = [1. for k in range(len(components))]

    if cal_flag:
        while exit_code:        # Loop over two functions
            # Try to perform drift and intensity adjustments
            exit_code, drifts, Icorr = GUIs.cal_gui(M.r, M.ppm, components, I, Icorr, exit_code, rav_flag)
            # Apply drift corrections to the peaks
            for k, ucorr in enumerate(drifts):
                for _, peak in comp_peaks[k].items():
                    peak.u += ucorr
            if exit_code == 0:  # You pressed the "save" button
                break
            else:   # You pressed the "edit" button
                idx = exit_code - 1     # Spectrum to edit peak-by-peak

            # Make a shallow copy
            peaks = deepcopy(comp_peaks[idx])
            # All spectra except the one to edit
            offset = np.sum([y for k, y in enumerate(components) if k != idx], axis=0)

            # Make correction
            comp_peaks[idx], Acorr = GUIs.edit_gui(M.r, M.ppm, peaks, acqus['t1'], acqus['SFO1'], acqus['o1p'], offset=offset, I=I, A0=Icorr[idx], rav_flag=rav_flag)
            # Make the correct integral of the spectrum
            for _, peak in comp_peaks[idx].items():
                peak.k *= Hs[idx]
            # Compute again the dictionary of kz.fit.Peak objects with the new parameters
            components = [np.sum([peak() for _, peak in dic_Peaks.items()], axis = 0) for dic_Peaks in comp_peaks]
            # Apply correction to the intensity
            Icorr[idx] = Icorr[idx] * Acorr
            
        # Check the match between integral and number of protons
        for k, comp in enumerate(comp_peaks):
            K_vals = [peak.k for _, peak in comp.items()]
            X, J = kz.misc.molfrac(K_vals)
            Hs[k] = int(round(J))

        # Write a new .fvf file for the calibrated components
        tmp_lims = max(M.ppm), min(M.ppm)   # Dummy limits
        for k, filename in enumerate(_spectra_dir):
            # <filename>.fvf becomes <filename>-cal.fvf
            base_name, extension = filename.rsplit('.', 1)
            new_filename = base_name + '-cal.' + extension
            print(filename, new_filename)
            # Write the new fvf file
            kz.fit.write_vf(new_filename, comp_peaks[k], tmp_lims, 1, header=True)#Hs[k])
        if filename == new_filename:
            print('Calibrated version of the .fvf files updated.')
        else:
            print('Calibrated version of the .fvf files saved.')
        I0str = ', '.join([f'{C:.4f}' for C in Icorr])
        print('\nInitial guess for the concentration computed.')
        print('Add the following lines to the input file:')
        print(f'I0\n{I0str}\n')

    # Select regions for performing the fit
    if lims is None:
        components = [np.sum([peak() for _, peak in dic_Peaks.items()], axis = 0) for dic_Peaks in comp_peaks]
        full_calc = np.sum([i*c for i, c in zip(Icorr, components)], axis=0)
        lims = GUIs.select_regions(M.ppm, M.r, full_calc)
        text_to_append = '\n'.join([f'{max(X):-7.3f}, {min(X):-7.3f}' for X in lims])
        print('Append the following text to your input file:\n\nFIT_LIMITS\n'+text_to_append+'\n')
        lims = [(max(X), min(X)) for X in lims]


    # Make the Spectr objects with only the peaks inside the regions
    comp_peaks_in = [{} for k, _ in enumerate(comp_peaks)]
    components = []
    for k, dic_Peaks in enumerate(comp_peaks):
        for key, peak in dic_Peaks.items():
            if is_in(peak.u, lims):
                peak.idx = key
                comp_peaks_in[k][key] = peak
        if len(comp_peaks_in[k]) == 0:
            components.append('Q')
        else:
            components.append(Spectr(acqus, *[peak for _, peak in comp_peaks_in[k].items()]))
        # Correct Hs
        Hs_in = np.sum([peak.k for _, peak in comp_peaks_in[k].items()])
        Hs[k] = round(Hs_in, 5)



    # Spectra that do not have peaks in the selected range
    def find_indices(list_to_check, item_to_find):
        return [idx for idx, value in enumerate(list_to_check) if value == item_to_find]
    missing = find_indices(Hs, 0)
    c_idx = [i for i, _ in enumerate(Hs) if i not in missing]
    # Remove them one by one
    if len(missing):
        I0 = [Icorr[k] for k, _ in enumerate(Hs) if k not in missing]
        while 0 in Hs:
            Hs.pop(Hs.index(0))
            components.pop(components.index('Q'))
        
        # Prompt message
        if len(missing) == 1:
            print(f'Component {", ".join([str(w+1) for w in missing])} has no peaks in the selected range.')
        else:
            print(f'Components {", ".join([str(w+1) for w in missing])} have no peaks in the selected range.')
    else:
        I0 = list(Icorr)

    return components, Hs, I0, lims, c_idx, I


