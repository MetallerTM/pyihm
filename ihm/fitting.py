#! /usr/bin/env python3

from klassez import *
from gen_param import main as gen_param

import lmfit as l

# Get the parameters from the other function
acqus, Lparam, param = gen_param()

# I put this because I didn't want to read it
N_spectra = 2

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
    peak_list = [fit.Peak(acqus,
        u=dic[f'u{i}'], 
        fwhm=dic[f's{i}'],
        k=dic[f'k{i}'],
        x_g=0, 
        phi=0,
        N=None,
        ) for i in peaks_idx[n]]
    # Compute the trace for each peak, then sum them up, finally multiply by the intensity
    spectra.append(dic['I'] * np.sum([peak() for peak in peak_list], axis=0))

# Sum the spectra to give the total fitting trace
total = np.sum(spectra, axis=0)
plt.plot(total)
plt.show()



    

    



