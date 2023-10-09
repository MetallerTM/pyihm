#! /usr/bin/env python3

import sys
import klassez as kz

filename = sys.argv[1]  # Spectrum
spect = sys.argv[2]     # Format

# Read the spectrum
S = kz.Spectrum_1D(filename, spect=spect)
# Do FT
S.process()
# Set "if 1" to phase correct
if 0:
    S.adjph()

# Create/read the initial guess for the deconvolution
S.F.iguess()
# Perform the fit...
S.F.dofit(  # ...with the following options:
        u_tol=0.2,          # variation on chemical shift /ppm
        f_tol=2,            # variation of FWHM /Hz
        vary_phase=False,   # Phase correction on the peaks
        vary_xg=False,      # Fraction of gaussianity
        )
# Save the figures for the fit
S.F.plot('result', show_res=True, res_offset=0.1)
