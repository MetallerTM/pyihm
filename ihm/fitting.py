#! /usr/bin/env python3

from klassez import *
from gen_param import main as gen_param
from gen_param import L2P

import lmfit as l

# Get the parameters from the other function
M, acqus, Lparam, param = gen_param()

# I put this because I didn't want to read it
N = M.r.shape[-1]
N_spectra = 2
exp = np.copy(M.r)
I = processing.integral(exp) * (2 * M.acqus['dw']) / 2

def calc_spectra(Lparam, param, N_spectra, acqus, N, use_L=True):
    # Convert normalized parameters into new parameters
    if use_L:
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
        peak_list = [fit.Peak(acqus,
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

#@cron
def f2min(Lparam, param, N_spectra, acqus, N, exp, I):
    Lparam['count'].set(Lparam['count'].value + 1)
    spectra = calc_spectra(Lparam, param, N_spectra, acqus, N)

    # Sum the spectra to give the total fitting trace
    total = np.sum(spectra, axis=0)
    residual = exp - I * total
    if Lparam['count'].value % 500 == 0: 
        print([f'{key}, {param[key].value}' for key in param if 'I' in key ])
        fig = plt.figure()
        plt.plot(exp, label='E')
        for k, s in enumerate(spectra):
            plt.plot(s, label=f'c{k+1}')
        plt.plot(total, label='F')
        plt.plot(residual, '--', label='R')
        plt.ylim(-max(exp), max(exp))
        plt.legend()
        plt.savefig(f'fit_{Lparam["count"].value}.tiff', dpi=600)
        plt.close()
    return np.sum(residual**2)

i_spectra = calc_spectra(Lparam, param, N_spectra, acqus, N)

plt.figure()
plt.plot(M.ppm, exp, lw=3)
plt.plot(M.ppm, I * np.sum(i_spectra, axis=0))
plt.figure()
plt.plot(M.ppm, exp, lw=3)
for s in i_spectra:
    plt.plot(M.ppm, I * s)
plt.figure()
plt.plot(M.ppm, exp - np.sum(i_spectra, axis=0) * I, '--')
plt.show()

param.pretty_print()
print('='*96)

count = 0
Lparam.add('count', value=0, vary=False)

minner = l.Minimizer(f2min, Lparam, fcn_args=(param, N_spectra, acqus, N, exp, I))
result = minner.minimize(method='nelder', max_nfev=10000, tol=1e-10)
#print(result.message, result.nfev)
print('')

param.pretty_print()
print('='*96)

Lopt = result.params

opt_spectra = calc_spectra(Lopt, param, N_spectra, acqus, N)

K = [f for key, f in param.valuesdict().items() if 'I' in key]
K_norm, _ = misc.molfrac(K)
print(K_norm)


plt.figure()
plt.plot(exp, lw=3)
plt.plot(I * np.sum(opt_spectra, axis=0))
plt.figure()
plt.plot(exp, lw=3)
for s in opt_spectra:
    plt.plot(I * s)
plt.figure()
plt.plot(exp - np.sum(opt_spectra, axis=0) * I, '--')
plt.show()


    

    



