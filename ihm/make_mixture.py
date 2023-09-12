#! /usr/bin/env python3

from klassez import *



comp_files = ['C_1.acqus', 'C_2.acqus']

mixing_perc = [0.1, 0.9]

S = []
for file in comp_files:
    tS = Spectrum_1D(file, isexp=False)
    tS.process()
    #tS.plot()
    S.append(tS)


if 1:
    M = Spectrum_1D(file, isexp=False)
    mfid = mixing_perc[0] * S[0].fid + mixing_perc[1] * S[1].fid
    M.filename = 'M'
    mfid += sim.noisegen(mfid.shape, M.acqus['o1'], M.acqus['t1'], s_n=0.1)
    M.fid = mfid
    M.process()
    M.plot()
    M.write_acqus()
    np.savetxt('M.fid', M.fid)

