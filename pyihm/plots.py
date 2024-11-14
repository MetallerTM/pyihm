#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import klassez as kz


def plot_iguess(ppm_scale, exp, total, components, lims=None, plims=None, X_label=r'$\delta$ /ppm', filename='fit', ext='tiff', dpi=600):
    """
    Makes the figure of the initial guess and saves it.
    -----------
    Parameters:
    - ppm_scale: 1darray
        PPM scale of the spectrum
    - exp: 1darray
        Mixture spectrum, real part
    - total: 1darray
        Fitting function
    - components: list of 1darray
        Spectra used as components, real part
    - lims: tuple or None
        Delimiters of the fitting region, in ppm. If None, the whole spectrum is used.
    - plims: list of slice or None
        Delimiters of the fitting windows, to cut the residuals
    - X_label: str
        Label for the X_axis
    - filename: str
        The name of the figure will be <filename>_iguess.<ext>
    - ext: str
        Format of the figures
    - dpi: int
        Resolution of the figures, in dots per inches
    """

    # Create the figure 
    fig = plt.figure()
    fig.set_size_inches(kz.figures.figsize_large)
    ax = fig.add_subplot(1,1,1)
    plt.subplots_adjust(left=0.10, right=0.95, bottom=0.10, top=0.95)

    # Make the plots
    #   of experimental spectrum
    kz.figures.ax1D(ax, ppm_scale, exp, c='k', lw=0.8, label='Experimental')
    #   of the total trace
    kz.figures.ax1D(ax, ppm_scale, total, c='tab:blue', lw=0.7, label='Fit')
    #   of each component
    for k, component in enumerate(components):
        trace = kz.figures.ax1D(ax, ppm_scale, component, lw=0.5, c=kz.COLORS[k+1], label=f'Comp. {k+1}')
        trace.set_linestyle('--')

    # Compute the residual and an offset to make it appear below the baseline of the spectra
    total = np.sum(components, axis=0)
    residuals = exp - total
    res_offset = 0.05 * (max(total)-min(total))
    # Plot it as a set of dots
    if plims:
        for k, X in enumerate(plims):
            r_plot = kz.figures.ax1D(ax, ppm_scale[X], residuals[X]-res_offset, c='tab:green', lw=0.7, label=k*'_'+'Residuals')
            r_plot.set(
                    ls='-',
                    marker='.',
                    markersize=0.6,
                    markeredgewidth=0.6,
                    fillstyle='full',
                    )
    

    # Adjust the x-scale according to lims
    if lims is None:
        kz.misc.pretty_scale(ax, (max(ppm_scale), min(ppm_scale)), axis='x')
    else:
        kz.misc.pretty_scale(ax, (max(lims), min(lims)), axis='x')
    # Adjust the y-scale
    kz.misc.pretty_scale(ax, kz.misc.get_ylim([exp, total, residuals-res_offset]), axis='y')
    # Set the label to the axes
    ax.set_xlabel(X_label)
    ax.set_ylabel('Intensity /a.u.')
    # Draw the legend
    ax.legend()
    # Adjust the fontsizes
    kz.misc.set_fontsizes(ax, 20)

    # Save the figure
    plt.savefig(f'{filename}_iguess.{ext}', dpi=dpi)
    plt.close()

def plot_output(ppm_scale, exp, total, components, lims=None, plims=None, X_label=r'$\delta$ /ppm', filename='fit', ext='tiff', dpi=600, windows=False):
    """
    Makes the figures of the final fitted spectrum and saves them. 
    Three figures are made: look at the fitting.main function documentation for details.
    -----------
    Parameters:
    - ppm_scale: 1darray
        PPM scale of the spectrum
    - exp: 1darray
        Mixture spectrum, real part
    - total: 1darray
        Fitting function
    - components: list of 1darray
        Spectra used as components, real part
    - lims: tuple or None
        Delimiters of the fitting region, in ppm. If None, the whole spectrum is used.
    - plims: list of slice or None
        Delimiters of the fitting windows, to cut the residuals
    - X_label: str
        Label for the X_axis
    - filename: str
        Root filename for the figures
    - ext: str
        Format of the figures
    - dpi: int
        Resolution of the figures, in dots per inches
    - windows: bool
        If True, saves a separate figure for each fitting window
    """

    ## FIRST FIGURE: experimental, total, residuals
    # Create the figure
    fig = plt.figure()
    fig.set_size_inches(kz.figures.figsize_large)
    ax = fig.add_subplot(1,1,1)
    plt.subplots_adjust(left=0.10, right=0.95, bottom=0.10, top=0.95)

    # Plot experimental spectrum and  total fitting function
    exp_plot = kz.figures.ax1D(ax, ppm_scale, exp, c='k', lw=0.8, label='Experimental')
    total_plot = kz.figures.ax1D(ax, ppm_scale, total, c='tab:blue', lw=0.7, label='Fit')

    # Compute the residual and an offset to make it appear below the baseline of the spectra
    residuals = exp - total
    res_offset = 0.05 * (max(total)-min(total))
    # Plot it as a set of dots
    if plims:
        for k, X in enumerate(plims):
            r_plot = kz.figures.ax1D(ax, ppm_scale[X], residuals[X]-res_offset, c='tab:green', lw=0.7, label=k*'_'+'Residuals')
            r_plot.set(
                    ls='-',
                    marker='.',
                    markersize=0.6,
                    markeredgewidth=0.6,
                    fillstyle='full',
                    )
    
    # Make the x-scale according to lims
    if lims is None:
        kz.misc.pretty_scale(ax, (max(ppm_scale), min(ppm_scale)), axis='x')
    else:
        kz.misc.pretty_scale(ax, (max(lims), min(lims)), axis='x')
    # Make the y-scale
    kz.misc.pretty_scale(ax, kz.misc.get_ylim([exp, total, residuals-res_offset]), axis='y')
    # Put the label to the axes
    ax.set_xlabel(X_label)
    ax.set_ylabel('Intensity /a.u.')
    # Draw the legend
    ax.legend()
    # Adjust the fontsizes
    kz.misc.set_fontsizes(ax, 20)

    # Save the figure
    plt.savefig(f'{filename}_total.{ext}', dpi=dpi)
    if windows:
        for k, X in enumerate(plims):
            exp_plot.set_data(ppm_scale[X], exp[X])
            total_plot.set_data(ppm_scale[X], total[X])
            r_plot.set_data(ppm_scale[X], residuals[X]-res_offset)
            kz.misc.pretty_scale(ax, (max(ppm_scale[X]), min(ppm_scale[X])), axis='x')
            kz.misc.pretty_scale(ax, kz.misc.get_ylim([exp[X], total[X], residuals[X]-res_offset]), axis='y')
            plt.savefig(f'{filename}_W{k+1}.{ext}', dpi=dpi)
    plt.close()


    ## SECOND FIGURE: experimental, total, components
    # Create the figure
    fig = plt.figure()
    fig.set_size_inches(kz.figures.figsize_large)
    ax = fig.add_subplot(1,1,1)
    plt.subplots_adjust(left=0.10, right=0.95, bottom=0.10, top=0.95)

    # Plot experimental spectrum and  total fitting function
    kz.figures.ax1D(ax, ppm_scale, exp, c='k', lw=0.8, label='Experimental')
    kz.figures.ax1D(ax, ppm_scale, total, c='tab:blue', lw=0.7, label='Fit')

    # Plot the components
    for k, component in enumerate(components):
        trace = kz.figures.ax1D(ax, ppm_scale, component, lw=0.5, c=kz.COLORS[k+1], label=f'Comp. {k+1}')
        trace.set_linestyle('--')

    # Make the x-scale according to lims
    if lims is None:
        kz.misc.pretty_scale(ax, (max(ppm_scale), min(ppm_scale)), axis='x')
    else:
        kz.misc.pretty_scale(ax, (max(lims), min(lims)), axis='x')
    # Make the y-scale
    kz.misc.pretty_scale(ax, kz.misc.get_ylim([exp, total]), axis='y')
    # Put the label to the axes
    ax.set_xlabel(X_label)
    ax.set_ylabel('Intensity /a.u.')
    # Draw the legend
    ax.legend()
    # Adjust the fontsizes
    kz.misc.set_fontsizes(ax, 20)

    # Save the figure
    plt.savefig(f'{filename}_wcomp.{ext}', dpi=dpi)
    plt.close()


    ## THIRD FIGURE: histogram of the residuals
    # Create the figure
    fig = plt.figure()
    fig.set_size_inches(kz.figures.figsize_large)
    ax = fig.add_subplot(1,1,1)
    plt.subplots_adjust(left=0.10, right=0.95, bottom=0.10, top=0.95)

    # Compute the number of bins
    # Formula: floor[ N / 10^(logN)] * 10^(logN - 1); N=length of residuals, logN=floor[log10(N)]
    # Example: 24623 points are bucketed into 2000 bins
    nbins = int(np.floor(residuals.shape[-1] / (10**np.floor(np.log10(residuals.shape[-1])))) * 10**(np.floor(np.log10(residuals.shape[-1])) - 1))

    # Make the histogram
    if plims:
        R_trim = np.concatenate([residuals[X] for X in plims])
        kz.fit.ax_histogram(ax, R_trim, nbins=nbins, density=True, f_lims=None, xlabel='Residuals', x_symm=True, barcolor='tab:green', fontsize=20)
    else:
        kz.fit.ax_histogram(ax, residuals, nbins=nbins, density=True, f_lims=None, xlabel='Residuals', x_symm=True, barcolor='tab:green', fontsize=20)

    # Adjust the fontsizes
    kz.misc.set_fontsizes(ax, 20)

    # Save the figure
    plt.savefig(f'{filename}_rhist.{ext}', dpi=dpi)
    plt.close()



def convergence_path(conv_path, filename='conv', ext='tiff', dpi=600):
    """
    Makes the figures of the final fitted spectrum and saves them. 
    Three figures are made: look at the fitting.main function documentation for details.
    -----------
    Parameters:
    - conv_path: str
        Path to the file of the convergence path
    - filename: str
        Filename of the final figure
    - ext: str
        Format of the figure
    - dpi: int
        Resolution of the figure, in dots per inches
    """
    # Read the file and unpack it
    arr = np.loadtxt(conv_path).T
    steps, target = arr[0], arr[1]

    # Create the figure
    fig = plt.figure()
    fig.set_size_inches(kz.figures.figsize_large)
    ax = fig.add_subplot(1,1,1)
    plt.subplots_adjust(left=0.10, top=0.95, bottom=0.10, right=0.95)

    # Make the plot
    ax.plot(steps, np.log10(target), '.', markersize=3, c='tab:blue')

    # Adjust xlim as matplotlib wants, but draw the scale only on the actual values
    xlim = ax.get_xlim()
    kz.misc.pretty_scale(ax, (min(steps), max(steps)), axis='x')
    ax.set_xlim(xlim)
    # Fancy shit
    kz.misc.pretty_scale(ax, ax.get_ylim(), axis='y')
    ax.set_xlabel('Iteration step')
    ax.set_ylabel('$\log$ target value')
    kz.misc.set_fontsizes(ax, 20)
    # Save the figure
    plt.savefig(f'{filename}.{ext}', dpi=dpi)
    plt.close()
    
