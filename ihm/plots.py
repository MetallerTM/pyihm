#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import klassez as kz


def plot_iguess(ppm_scale, exp, total, components, lims=None, X_label=r'$\delta$ /ppm', filename='', ext='tiff', dpi=600):

    fig = plt.figure()
    fig.set_size_inches(kz.figures.figsize_large)
    ax = fig.add_subplot(1,1,1)
    plt.subplots_adjust(left=0.10, right=0.95, bottom=0.10, top=0.95)

    kz.figures.ax1D(ax, ppm_scale, exp, c='k', lw=0.8, label='Experimental')
    kz.figures.ax1D(ax, ppm_scale, total, c='tab:blue', lw=0.7, label='Fit')
    for k, component in enumerate(components):
        trace = kz.figures.ax1D(ax, ppm_scale, component, lw=0.5, c=kz.COLORS[k+1], label=f'Comp. {k+1}')
        trace.set_linestyle('--')

    if lims is None:
        kz.misc.pretty_scale(ax, (max(ppm_scale), min(ppm_scale)), axis='x')
    else:
        kz.misc.pretty_scale(ax, (max(lims), min(lims)), axis='x')
    kz.misc.pretty_scale(ax, kz.misc.get_ylim(exp), axis='y')
    ax.set_xlabel(X_label)
    ax.set_ylabel('Intensity /a.u.')
    ax.legend()
    kz.misc.set_fontsizes(ax, 16)

    plt.savefig(f'{filename}_iguess.{ext}', dpi=dpi)
    plt.close()


