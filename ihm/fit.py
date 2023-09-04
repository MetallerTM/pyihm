#! /usr/bin/env python3

from klassez import *
import lmfit as l

param = l.Parameters()

bds = {
        'u_big':    0.2,    #ppm
        'u_small':  0.01,   #ppm
        's': 0.01,  #%
        'k': 0.01,  #%
        }

u = l.Parameter(
        name = 'u',
        value = 0,
        min = 0 - bds['u_big'],
        max = 0 + bds['u_big'],
        )

def P2L(P):
    L = l.Parameter(
            name = f'L{P.name}',
            value = (P.value - P.min) / (P.max - P.min),
            min = 0,
            max = 1,
            )
    return L

def L2P(L, Xmin, Xmax):
    name = f'{L.name}'.replace('L', '', 1)
    value = L.value * (Xmax - Xmin) + Xmin
    return name, value




