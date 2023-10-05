#! /usr/bin/env python3

import numpy as np
import klassez as kz

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
                    'x_g': peak.x_g,
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


def main(M, spectra_dir):
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
    ----------
    Returns:
    - collections: list of Spectr objects
        Spectra of pure components, treated as collections of peaks.
    """
    # Get "structural" parameters from M
    acqus = dict(M.acqus)
    N = M.r.shape[-1]       # Number of points for zero-filling
    ## Gather all the peaks
    components = [] # Whole spectra
    # Collect the parameters of the peaks
    spectra_peaks = [kz.fit.read_vf(file) for file in spectra_dir]
    for all_peaks in spectra_peaks: # Unpacks the fitting regions
        whole_spectrum = []   # Create empty list of components
        for region_peaks in all_peaks:      # Unpack the peaks in a given region
            # Remove total intensity and fitting window
            I = region_peaks.pop('I')
            region_peaks.pop('limits')
            peaks = []      # Empty list
            for key in sorted(region_peaks.keys()): # Iterate on the peak index
                p = dict(region_peaks[key]) # Alias, shortcut
                # Create the kz.fit.Peak object and append it to the peaks list. Use the ABSOLUTE intensities in order to not mess up with different windows!
                peaks.append(kz.fit.Peak(acqus, u=p['u'], fwhm=p['fwhm'], k=I*p['k'], x_g=p['x_g'], phi=0, N=N, group=p['group']))
                # Add the peak index as "floating" attribute
                peaks[-1].idx = key
            # Once all the peaks in a given region have been generated, store them in the list 
            whole_spectrum.extend(peaks)

        ## Normalize the intensity values
        # Get the absolute values
        K_vals = [p.par()['k'] for p in whole_spectrum]
        # Normalize them
        K_norm, _ = kz.misc.molfrac(K_vals)
        # Put the new ones
        for p, k in zip(whole_spectrum, K_norm):
            p.k = k

        # At the end, generate the Spectr object and add it to a list
        components.append(Spectr(acqus, *whole_spectrum))

    return components
