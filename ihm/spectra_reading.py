#! /usr/bin/env python3

from klassez import *

class Multiplet:
    def __init__(self, acqus, *peaks):
        self.acqus = acqus
        self.peaks = {}
        for peak in peaks:
            self.peaks[peak.idx] = peak

        self.U = np.mean([p.u for _, p in self.peaks.items()])
        self.u_off = {key: p.u - self.U for key, p in self.peaks.items()}

    def __call__(self):
        trace = np.zeros(self.acqus['TD'])
        for key, peak in self.peaks.items():
            self.peaks[key].u = self.U + self.u_off[key]
            trace += peak()
        return trace


class Spectr:
    def __init__(self, acqus, *peaks):
        self.acqus = acqus
        self.peaks = {}
        for peak in peaks:
            self.peaks[peak.idx] = peak

        all_groups = {key: p.group for key, p in self.peaks.items()}
        self.unique_groups = sorted(list(set([g for _, g in all_groups.items()])))
        self.p_collections = {}
        for g in self.unique_groups:
            keys = [key for key, item in all_groups.items() if item == g]
            if g == 0:
                self.p_collections[0] = [Multiplet(self.acqus, self.peaks[key]) for key in keys]
            else:
                self.p_collections[g] = [Multiplet(self.acqus, *[self.peaks[key] for key in keys])]
        self.total = self.calc_total()

    def calc_total(self):
        total = np.zeros(acqus['TD'])
        for g in self.unique_groups:
            for s in self.p_collections[g]:
                total += s()
        return total

    def __call__(self, I=1):
        total = I * self.calc_total()
        return total






# Load the acqus dictionary of the (mixture) spectrum
acqus = sim.load_sim_1D('Sp_1.acqus')
# Create list of peaks files
spectra_dir = ['Sp_1.fvf', 'Sp_1.fvf']


## Gather all the peaks
components = [] # Whole spectra
# Collect the parameters of the peaks
spectra_peaks = [fit.read_vf(file) for file in spectra_dir]
for all_peaks in spectra_peaks: # Unpacks the fitting regions
    all_spectrum = []   # Create empty list of components
    for region_peaks in all_peaks:      # Unpack the peaks in a given region
        # Remove total intensity and fitting window
        region_peaks.pop('I')
        region_peaks.pop('limits')
        peaks = []      # Empty list
        for key in sorted(region_peaks.keys()): # Iterate on the peak index
            p = dict(region_peaks[key]) # Alias, shortcut
            # Create the fit.Peak object and append it to the peaks list
            # TODO cambiare N per matchare le dimensioni dello spettro della miscela
            peaks.append(fit.Peak(acqus, u=p['u'], fwhm=p['fwhm'], k=p['k'], x_g=p['x_g'], phi=0, N=None, group=p['group']))
            # Add the peak index as "floating" attribute
            peaks[-1].idx = key
        # Once all the peaks in a given region have been generated, store them in the list 
        all_spectrum.extend(peaks)

    # At the end, generate the Spectr object and add it to a list
    components.append(Spectr(acqus, *all_spectrum))

for k, S in enumerate(components):
    plt.plot(S() + 10*k)
plt.show()
pass
