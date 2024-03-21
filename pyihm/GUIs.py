#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, SpanSelector, RadioButtons, Slider
import klassez as kz

from .fit_mixture import calc_spectra

def select_regions(ppm_scale, spectrum, full_calc):
    """
    Interactively select the slices that will be used in the fitting routine.
    ---------------
    Parameters:
    - ppm_scale: 1darray
        ppm scale of the spectrum
    - spectrum: 1darray
        Spectrum of the mixture
    - full_calc: 1darray
        Spectrum of the initial guess, with all the peaks in total
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
    ax.plot(ppm_scale, S, c='tab:blue', lw=1.2, label='Mixture')
    if isinstance(full_calc, np.ndarray):
        full_calc_norm = full_calc / max(full_calc) * max(S)
        ax.plot(ppm_scale, full_calc_norm, c='r', lw=0.7, label='Initial guess')

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


def cal_gui(ppm_scale, exp, param, N_spectra, acqus, N, I):
    """
    Corrects the chemical shifts and the intensities of the spectra to be employed during the fit.
    ----------
    Parameters:
    - ppm_scale: 1darray
        Chemical shift scale of the spectrum
    - exp: 1darray
        Experimental spectrum
    - param: lmfit.Parameters object
        Parameters of the fit, as generated by gen_param
    - N_spectra: int
        Number of components of the mixture
    - acqus: dict
        Dictionary of acquisition parameters
    - N: int
        Number of points that the final calculated spectum should have
    - I: float
        Intensity correction for the experimental spectrum
    ------------
    Returns:
    - param: lmfit.Parameters object
        Updated parameters
    """
    # Calculate the spectra of the components of the mixture
    i_spectra = calc_spectra(param, N_spectra, acqus, N)

    comp_idx = [eval(p.split('_I')[0].replace('S', '')) for p in param if '_I' in p]

    # Initialize the parameters to be modified interactively
    roll_n = [0 for w in range(N_spectra)]      # shift in points
    I_s = [1 for w in range(N_spectra)]         # intensity factor
    sens_I = 0.1                                # sensitivity for intensity
    sens_u = 0.01                               # sensitivity for shift, in ppm

    pt_ppm = kz.misc.calcres(ppm_scale)         # resolution of the scale in ppm per point
    # Make sure the calibration sensitivity is exactly a mutiple of a point shift
    sens_n = int(round(sens_u / pt_ppm))    
    sens_u = sens_n * pt_ppm

    # Make the figure panel and adjust the borders
    fig = plt.figure()
    fig.set_size_inches(15,8)
    ax = fig.add_subplot(1,1,1)
    plt.subplots_adjust(left=0.075, right=0.8, bottom=0.1, top=0.95)

    # Make the boxes for the widgets
    box_slider = plt.axes([0.820, 0.10, 0.005, 0.85])       # Box for spectrum selector
    box_up = plt.axes([0.85, 0.90, 0.05, 0.05])             # Box for increase sensitivity button
    box_down = plt.axes([0.925, 0.90, 0.05, 0.05])          # Box for decrease sensitivity button
    box_radio = plt.axes([0.85, 0.70, 0.125, 0.175])        # Box for radiobuttons

    prompt = plt.axes([0.85, 0.10, 0.125, 0.575])           # Box for writing the information on the modified values
    prompt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)    # Remove the labels from the spines
    prompt.text(0.05, 0.975, f'{"#":>2s}: {"cal":>10s}, {"I":>10s}', ha='left', va='center', transform=prompt.transAxes)    # Add header

    ## WIDGETS
    #   Selector slider
    slider = Slider(box_slider, 'Spectrum', valmin=1, valmax=N_spectra, valinit=1, valstep=1, orientation='vertical')
    #   Radiobuttons
    radio = RadioButtons(box_radio, ['DRIFT', 'INTENS'])
    #   Sensitivity buttons
    up_button = Button(box_up, r'$\uparrow$', hovercolor='0.975')
    down_button = Button(box_down, r'$\downarrow$', hovercolor='0.975')

    ## SLOTS
    def set_line_active(act):
        """ Active spectrum becomes red, all the others blue """
        # act = index of the active spectrum
        for k, line in enumerate(y_lines):
            if k == act:
                line.set_color('tab:red')
            else:
                line.set_color('tab:blue')
        plt.draw()

    def move_drift(sign, act):
        """ Roll the active spectrum and updates the values """
        nonlocal i_spectra, roll_n
        # the minus is because increasing the chemical shift should roll towards left
        i_spectra[act] = np.roll(I_s[act] * i_spectra[act], -sign*sens_n)
        y_lines[act].set_ydata(i_spectra[act])
        roll_n[act] += sign*sens_n  # But here it is of the correct sign, +
        plt.draw()

    def move_intens(sign, act):
        """ Adjust the intensity of the active spectrum and updates the values """
        nonlocal I_s
        I_s[act] += sign*sens_I
        y_lines[act].set_ydata(I_s[act]*i_spectra[act])
        plt.draw()

    def mouse_scroll(event):
        """ Mouse scroll slot """
        # Get active spectrum
        act = int(slider.val) - 1
        # Determine the sign of the rolling
        if event.button == 'up':
            sign = 1
        elif event.button == 'down':
            sign = -1
        else:
            return

        # Discriminate the behavior according to the radiobuttons
        if radio.value_selected == 'DRIFT':
            move_drift(sign, act)
        elif radio.value_selected == 'INTENS':
            move_intens(sign, act)
        # Update the text
        set_prompt_text()

    def slot_slider(event):
        """ Changes the active spectrum """
        act = int(slider.val) - 1
        set_line_active(act)

    def up_sens(event):
        """ Increase the sensitivity """
        if radio.value_selected == 'DRIFT':
            nonlocal sens_n, sens_u
            sens_u *= 2
            # Make sure that sens_u is a multiple of a point shift
            sens_n = int(round(sens_u / pt_ppm))
            sens_u = sens_n * pt_ppm
        elif radio.value_selected == 'INTENS':
            nonlocal sens_I
            sens_I *= 2
        # Update the text
        set_sens_text()

    def down_sens(event):
        """ Increase the sensitivity """
        if radio.value_selected == 'DRIFT':
            nonlocal sens_n, sens_u
            sens_u /= 2
            # Make sure that sens_u is a multiple of a point shift
            sens_n = int(round(sens_u / pt_ppm))
            sens_u = sens_n * pt_ppm
        elif radio.value_selected == 'INTENS':
            nonlocal sens_I
            sens_I /= 2
        # Update the text
        set_sens_text()

    def set_sens_text():
        """ Update the sensitivity text """
        text_sens_u.set_text(f'$\pm${sens_u:10.3g}')
        text_sens_I.set_text(f'$\pm${sens_I:10.3g}')
        plt.draw()

    def set_prompt_text():
        """ Update the values text """
        for k in range(N_spectra):
            text_prompt[k].set_text(
                    f'{comp_idx[k]:>2.0f}: {roll_n[k]*pt_ppm:-10.4f}, {I_s[k]:10.4f}'
                    )
        plt.draw()

    def key_bindings(event):
        """ Key-binding for keyboard shortcuts """
        key = event.key
        act = int(slider.val - 1)
        if key == 'left':
            move_drift(+1, act)
        if key == 'right':
            move_drift(-1, act)
        if key == 'up':
            move_intens(+1, act)
        if key == 'down':
            move_intens(-1, act)
        if key == 'pageup':
            newval = slider.val + 1
            if newval <= N_spectra:
                slider.set_val(newval)
        if key == 'pagedown':
            newval = slider.val - 1
            if newval >= 1:
                slider.set_val(newval)
        if key == '>':
            up_sens(0)
        if key == '<':
            down_sens(0)
        set_prompt_text()

    ## PLOT
    # Experimental spectrum
    ax.plot(ppm_scale, exp/I, c='k', lw=1)

    # Calculated spectra 
    y_lines = []    # Placeholder
    for k, y in enumerate(i_spectra):
        # One line per spectrum
        line, = ax.plot(ppm_scale, y, c='tab:blue', lw=0.8)
        y_lines.append(line)
    # First one set as active
    y_lines[0].set_color('tab:red')

    # Placeholder for sensitivity text
    text_sens_u = box_radio.text(0.975, radio.labels[0].get_position()[-1], f'', ha='right', va='center', transform=box_radio.transAxes)
    text_sens_I = box_radio.text(0.975, radio.labels[1].get_position()[-1], f'', ha='right', va='center', transform=box_radio.transAxes)
    set_sens_text()

    # Placeholder for values text
    text_prompt = []    # Placeholder for Text instances
    y_coord = 0.975     # Where the title ends
    for k in range(N_spectra):
        y_coord -= 0.05 # Move down
        text_prompt.append(prompt.text(
            0.05, y_coord, '',
            ha='left', va='center', transform=prompt.transAxes))
    set_prompt_text()

    # Fancy stuff for axes and fontsizes
    ax.set_title(r'Calibration for field drift and intensity')
    ax.set_xlabel(r'$\delta\,$ /ppm')
    kz.misc.pretty_scale(ax, ax.get_xlim()[::-1], 'x')
    kz.misc.pretty_scale(ax, ax.get_ylim(), 'y')
    kz.misc.set_fontsizes(ax, 16)

    ## CONNECT WIDGETS TO SLOTS
    slider.on_changed(slot_slider)
    up_button.on_clicked(up_sens)
    down_button.on_clicked(down_sens)
    fig.canvas.mpl_connect('scroll_event', mouse_scroll)
    fig.canvas.mpl_connect('key_press_event', key_bindings)

    # Start event loop
    plt.show()

    # Get the optimized values
    drifts = np.array(roll_n) * pt_ppm  # Compute the calibration values in ppm
    Icorr = np.array(I_s)               # Convert to array
    # Adjust the parameters
    for k in range(N_spectra):
        param[f'S{comp_idx[k]}_I'].value *= Icorr[k]    # Intensity
        for p in param:
            if f'S{k+1}_' in p and ('U' in p or 'u' in p):  # Chemical shifts
                param[p].value += drifts[k]

    return param
