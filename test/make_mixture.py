#! /usr/bin/env python3

from klassez import *



comp_files = ['C_1.acqus', 'C_2.acqus', 'C_3.acqus']

mixing_perc = [0.2, 0.5, 0.3]

S = []
for file in comp_files:
    tS = Spectrum_1D(file, isexp=False)
    tS.process()
    #tS.plot()
    S.append(tS)

def lc(C, S):
    mix = np.zeros(max([len(s) for s in S]), dtype='complex128')
    for c, s in zip(C, S):
        mix += c * s
    return mix


if 1:
    M = Spectrum_1D(file, isexp=False)
    mfid = lc(mixing_perc, [s.fid for s in S])
    M.filename = 'M'
    mfid += sim.noisegen(mfid.shape, M.acqus['o1'], M.acqus['t1'], s_n=0.05)
    M.fid = mfid
    M.process()
    M.plot()
    M.write_acqus()
    M.procs['zf'] = 16384
    M.process()
    np.savetxt('M.fid', M.fid)
    np.savetxt('M.r', M.r)

