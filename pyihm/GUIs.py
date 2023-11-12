#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, SpanSelector
import klassez as kz


def select_regions(ppm_scale, spectrum):
    """
    Interactively select the slices that will be used in the fitting routine.
    ---------------
    Parameters:
    - ppm_scale: 1darray
        ppm scale of the spectrum
    - spectrum: 1darray
        Spectrum of the mixture
    --------------
    Returns:
    - regions: list of tuple
        Limits, in ppm
    """
    def merge_regions(regions):
        """
        Merge superimposing regions.
        ---------
        Parameters:
        - regions: list of tuple
            List of regions to be merged
        ---------
        Returns:
        - sort_reg: list of tuple
            Sorted and merged regions
        """
        def is_in(x, B):
            """
            Check if value is inside a given interval, i.e. if the following condition is satisfied:
                min(B) <= x && x <= max(B)
            ---------
            Parameters:
            - x: float
                Value to check
            - B: sequence
                Interval of interest.
            ---------
            Returns:
            - isin: bool
                True if the condition is satisfied, False otherwise
            """
            if min(B) <= x and x <= max(B):
                return True
            else:
                return False

        def check_format(regions):
            """ 
            Check if the tuples in the regions list are either all (max, min) or all (min, max).
            If they are in mixed order, it raises an error.
            ----------
            Parameters:
            - regions: list of tuple
                List to be checked
            ----------
            Returns:
            - rev: bool
                True = (max, min), False = (min, max)
            """
            rev = None
            for k, region in enumerate(regions):
                if region[0] > region[1]:
                    rev = True
                else:
                    rev = False
                if k == 0:
                    flag = rev
                assert flag == rev, 'The regions do not have all the same format.'
            return rev

        # Make all tuples in regions to be (max, min)
        corr_reg = [(max(X), min(X)) for X in regions]
        # Sort them
        sort_reg = sorted(corr_reg, reverse=check_format(corr_reg))

        # Merging loop
        k = 0   
        while k < len(sort_reg)-1:  # Which means we can always find sort_reg[k+1]
            r1 = sort_reg[k]        # "active" region
            r2 = sort_reg[k+1]      # next region
            if is_in(r2[0], r1):    # if the left border of next region is inside the active region:
                sort_reg[k] = (r1[0], r2[1])    # merge them, replacing the active region
                sort_reg.pop(k+1)               # remove the next region, as it would be duplicated now
            else:                   # Everything is good as is, advance the pointer
                k += 1

        return sort_reg


    ## SLOTS
    def onselect(xmin, xmax):
        """ Print the borders of the span selector """
        span.set_visible(True)
        text = f'{xmax:-6.3f}:{xmin:-6.3f}'
        span_limits.set_text(text)
        plt.draw()
        pass

    def add(event):
        """ Add to the list """
        nonlocal return_list
        # Do nothing if the span selector is invisible/not set
        if len(span_limits.get_text()) == 0 or span.get_visible is False:
            return
        # Get the current limits rounded to the third decimal figure
        lims = np.around(span.extents, 3)
        # Append these values to the final list, in the correct order
        return_list.append((max(lims), min(lims)))
        # Draw a permanent span 
        ax.axvspan(*lims, facecolor='tab:green', edgecolor='g', alpha=0.2)
        # Write the limits in the output box
        text = f'{max(lims):-6.3f}:{min(lims):-6.3f}'
        output = output_text.get_text()
        output_text.set_text('\n'.join([output, text]))
        # Reset the interactive text (the red one)
        span_limits.set_text('')
        # Turn the span selector invisible
        span.set_visible(False)
        plt.draw()

    def save(event):
        plt.close()
        regions = merge_regions(return_list)
        return regions

    #----------------------------------------------------------------------------------------------------

    # Shallow copy of the spectrum
    S = np.copy(spectrum.real)
    # Placeholder for return values
    return_list = []

    # Make the figure
    fig = plt.figure()
    fig.set_size_inches(kz.figures.figsize_large)
    ax = fig.add_subplot(1,1,1)
    plt.subplots_adjust(left=0.10, right=0.80, top=0.90, bottom=0.10)

    # Make boxes for widgets
    output_box = plt.axes([0.875, 0.100, 0.10, 0.70])
    output_box.set_facecolor('0.985')       # Grey background
    output_box.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    add_box = plt.axes([0.815, 0.820, 0.05, 0.06])
    save_box = plt.axes([0.815, 0.100, 0.05, 0.08])

    # Make widgets
    add_button = Button(add_box, 'ADD', hovercolor='0.975')
    save_button = Button(save_box, 'SAVE\nAND\nEXIT', hovercolor='0.975')

    # Plot the spectrum
    ax.plot(ppm_scale, S, c='tab:blue', lw=0.8)

    # Add axes labels
    ax.set_xlabel(r'$\delta\,$ /ppm')
    ax.set_ylabel(r'Intensity /a.u.')

    # Fancy adjustments
    kz.misc.pretty_scale(ax, (max(ppm_scale), min(ppm_scale)), 'x')
    kz.misc.pretty_scale(ax, ax.get_ylim(), 'y')
    kz.misc.mathformat(ax)
    kz.misc.set_fontsizes(ax, 20)

    # Declare span selector
    span = SpanSelector(
            ax,
            onselect,
            'horizontal', 
            useblit=True,
            props=dict(alpha=0.2, facecolor='tab:red'),
            interactive=True,
            drag_from_anywhere=True,
            )

    # Connect widgets to the slots
    add_button.on_clicked(add)
    save_button.on_clicked(save)

    # Make output text
    span_limits = plt.text(0.925, 0.85, '', ha='center', va='center', transform=fig.transFigure, fontsize=11, color='tab:red') 
    output_text = output_box.text(0.5, 1.025, '\n', ha='center', va='top', color='tab:green', fontsize=11)

    plt.show()

    # Merge superimposing regions and sort them
    regions = merge_regions(return_list)
    return regions

