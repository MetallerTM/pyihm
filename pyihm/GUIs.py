#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, SpanSelector, RadioButtons, Slider, TextBox
import klassez as kz
from copy import deepcopy

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
    ax.plot(ppm_scale, S, c='k', lw=1.2, label='Mixture')
    if isinstance(full_calc, np.ndarray):
        full_calc_norm = full_calc / max(full_calc) * max(S)
        ax.plot(ppm_scale, full_calc_norm, c='tab:blue', lw=0.7, label='Initial guess')

    ax.set_title(r'Drag the mouse to draw a region. Press "ADD" to add it to the list.')
    # Add axes labels
    ax.set_xlabel(r'$\delta\,$ /ppm')
    ax.set_ylabel(r'Intensity /a.u.')

    # Fancy adjustments
    kz.misc.pretty_scale(ax, (max(ppm_scale), min(ppm_scale)), 'x')
    kz.misc.pretty_scale(ax, ax.get_ylim(), 'y')
    kz.misc.mathformat(ax)
    kz.misc.set_fontsizes(ax, 14)

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


def cal_gui(exp, ppm_scale, components, I, prev_Icorr=None, init_active=1, rav_flag=False):
    """
    Corrects the chemical shifts and the intensities of the spectra to be employed during the fit.
    Works together with edit_gui, that allows to break up a spectrum in single components in order to adjust them.
    ----------
    Parameters:
    - exp: 1darray
        Experimental spectrum
    - ppm_scale: 1darray
        Chemical shift scale of the spectrum
    - components: list of 1darray
        Spectra to calibrate
    - I: float
        Intensity correction for the experimental spectrum
    - prev_Icorr: list
        Starting concentration to be further edited
    - init_active: int
        Initial spectrum to be activated
    - rav_flag: bool
        Activates colorblind paelette
    ------------
    Returns:
    - exit_code: int
        If 0, there is nothing to edit further. If not, it is the number of the component to be edited with edit_gui. The python-ish index is therefore (exit_code - 1)
    - drifts: 1darray
        Correction for the chemical shift correction of each spectrum, in ppm
    - Icorr
        Correction for the intensity of each spectrum
    """
    # Initialize variables and resets
    if rav_flag:
        v_l = kz.misc.cmap2list(kz.CM['viridis'], N=10, start=0, end=1)
        colors = {'exp': v_l[2], 'act': v_l[7], 'nonact':v_l[0], 'total':'m'}
    else:
        colors = {'exp': 'k', 'act':'tab:red', 'nonact':'tab:blue', 'total':'b'}


    _components = deepcopy(components)          # backup
    N_spectra = len(components)                 # Number of spectra
    comp_idx = [w+1 for w in range(N_spectra)]  # Indices of the spectra as 1, 2, 3...
    # values to be returned
    drifts = None           # chemical shift correction /ppm
    exit_code = None        

    # Initialize the parameters to be modified interactively
    roll_n = [0 for w in range(N_spectra)]      # shift in points
    if prev_Icorr is None:
        I_s = [1 for w in range(N_spectra)]         # intensity factor
    else:
        I_s = deepcopy(prev_Icorr)         # intensity factor
    sens_I = 0.1                                # sensitivity for intensity
    sens_u = 0.01                               # sensitivity for shift, in ppm

    pt_ppm = kz.misc.calcres(ppm_scale)         # resolution of the scale in ppm per point
    # Make sure the calibration sensitivity is exactly a mutiple of a point shift
    sens_n = int(round(sens_u / pt_ppm))    
    sens_u = sens_n * pt_ppm

    # Make the figure panel and adjust the borders
    fig = plt.figure('Calibration for drift and intensity')
    fig.set_size_inches(15,8)
    ax = fig.add_subplot(1,1,1)
    plt.subplots_adjust(left=0.075, right=0.8, bottom=0.1, top=0.95)

    # Make the boxes for the widgets
    box_slider = plt.axes([0.820, 0.10, 0.005, 0.85])       # Box for spectrum selector
    box_up = plt.axes([0.85, 0.90, 0.05, 0.05])             # Box for increase sensitivity button
    box_down = plt.axes([0.925, 0.90, 0.05, 0.05])          # Box for decrease sensitivity button
    box_radio = plt.axes([0.85, 0.80, 0.125, 0.075])        # Box for radiobuttons
    save_box = plt.axes([0.85, 0.100, 0.06, 0.05])          # Box for 'save' button
    edit_box = plt.axes([0.915, 0.100, 0.06, 0.05])         # Box for 'edit' button

    prompt = plt.axes([0.85, 0.175, 0.125, 0.600])           # Box for writing the information on the modified values
    prompt.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)    # Remove the labels from the spines
    prompt.text(0.05, 0.975, f'{"#":>2s}: {"cal":>10s}, {"I":>10s}', ha='left', va='center', transform=prompt.transAxes)    # Add header

    ## WIDGETS
    #   Selector slider
    slider = Slider(box_slider, 'Spectrum', valmin=1, valmax=N_spectra, valinit=init_active, valstep=1, orientation='vertical')
    #   Radiobuttons
    radio = RadioButtons(box_radio, ['DRIFT', 'INTENS'])
    #   Sensitivity buttons
    up_button = Button(box_up, r'$\uparrow$', hovercolor='0.975')
    down_button = Button(box_down, r'$\downarrow$', hovercolor='0.975')
    # Save button
    save_button = Button(save_box, 'SAVE', hovercolor='0.975')
    # Edit button
    edit_button = Button(edit_box, 'EDIT', hovercolor='0.975')

    ## SLOTS
    def set_line_active(act):
        """ Active spectrum becomes red, all the others blue """
        # act = index of the active spectrum
        for k, line in enumerate(y_lines):
            if k == act:
                line.set_color(colors['act'])
                line.set_lw(2)
                line.set_zorder(10)
            else:
                line.set_color(colors['nonact'])
                line.set_lw(0.8)
                line.set_zorder(3)
        plt.draw()

    def move_drift(sign, act):
        """ Roll the active spectrum and updates the values """
        nonlocal roll_n
        # the minus is because increasing the chemical shift should roll towards left
        roll_n[act] += sign*sens_n  # But here it is of the correct sign, +
        redraw(act)

    def move_intens(sign, act):
        """ Adjust the intensity of the active spectrum and updates the values """
        nonlocal I_s
        I_s[act] += sign*sens_I
        redraw(act)

    def redraw(act):
        """ Updates the plot according to the current values """
        nonlocal components
        components[act] = I_s[act]*np.roll(_components[act], -roll_n[act])
        y_lines[act].set_ydata(components[act])
        total = np.sum(components, axis=0)
        total_line.set_ydata(total)
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
            sens_n = max(1, int(round(sens_u / pt_ppm)))
            sens_u = sens_n * pt_ppm
        elif radio.value_selected == 'INTENS':
            nonlocal sens_I
            sens_I /= 2
        # Update the text
        set_sens_text()

    def set_sens_text():
        """ Update the sensitivity text """
        text_sens_u.set_text(r'$\pm$'+f'{sens_u:10.3g}')
        text_sens_I.set_text(r'$\pm$'+f'{sens_I:10.3g}')
        plt.draw()

    def set_prompt_text():
        """ Update the values text """
        for k in range(N_spectra):
            text_prompt[k].set_text(
                    f'{comp_idx[k]:>2.0f}: {roll_n[k]*pt_ppm:-10.4f}, {I_s[k]:10.4f}'
                    )
        plt.draw()

    def compute_return_values():
        # Get the optimized values
        drifts = np.array(roll_n) * pt_ppm  # Compute the calibration values in ppm
        Icorr = np.array(I_s)               # Convert to array
        return drifts, Icorr

    def save(event):
        """ Close the figure and returns the values """
        nonlocal drifts, Icorr, exit_code
        drifts, Icorr = compute_return_values()
        exit_code = 0   # close and never return
        plt.close()

    def edit(event):
        """ Close the figure and returns the values """
        nonlocal drifts, Icorr, exit_code
        drifts, Icorr = compute_return_values()
        exit_code = int(slider.val) # returns the spectrum to edit with edit_gui
        plt.close()

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
        if key == 't':
            total_line.set_visible(not total_line.get_visible())
        set_prompt_text()

    ## PLOT
    # Experimental spectrum
    ax.plot(ppm_scale, exp/I, c=colors['exp'], lw=1)

    # Calculated spectra 
    y_lines = []    # Placeholder
    for k, y in enumerate(components):
        # One line per spectrum
        line, = ax.plot(ppm_scale, I_s[k]*y, c=colors['nonact'], lw=0.8)
        y_lines.append(line)
    # First one set as active
    y_lines[init_active-1].set_color(colors['act'])
    y_lines[init_active-1].set_lw(2)
    y_lines[init_active-1].set_zorder(10)
    total_line, = ax.plot(ppm_scale, np.sum([i*y for i, y in zip(I_s, components)], axis=0), c=colors['total'], lw=0.4, zorder=2)
    total_line.set_visible(False)

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
    ax.set_title(r'Use the mouse scroll to edit the red spectrum. Press "t" to activate total trace.')
    ax.set_xlabel(r'$\delta\,$ /ppm')
    kz.misc.pretty_scale(ax, ax.get_xlim()[::-1], 'x')
    kz.misc.pretty_scale(ax, ax.get_ylim(), 'y')
    kz.misc.set_fontsizes(ax, 14)

    ## CONNECT WIDGETS TO SLOTS
    slider.on_changed(slot_slider)
    up_button.on_clicked(up_sens)
    down_button.on_clicked(down_sens)
    save_button.on_clicked(save)
    edit_button.on_clicked(edit)
    fig.canvas.mpl_connect('scroll_event', mouse_scroll)
    fig.canvas.mpl_connect('key_press_event', key_bindings)

    # Start event loop
    plt.show()

    if exit_code is None:   # The figure was closed without pressing any button
        # Behaves as "save"
        drifts, Icorr = compute_return_values()
        exit_code = 0

    return exit_code, drifts, Icorr

def edit_gui(exp, ppm_scale, peaks, t_AQ, SFO1, o1p, offset=None, I=1, A0=None, rav_flag=False):
    """
    Opens a GUI for the interactive edit of a spectrum, peak-by-peak. The usage resembles the one for the manual computation of the initial guess for the fit in KLASSEZ.
    At the end, all the relative intensities of the peaks sum up to 1. You need to restore the total intensity outside this function.
    For this reason, it is not possible to add new peaks, or remove already existing ones. For this latter option, consider to move them in an empty region of the spectrum, and exclude them with select_regions.
    -------------
    Parameters:
    - exp: 1darray
        Experimental spectrum
    - ppm_scale: 1darray
        PPM scale of the spectrum
    - peaks: dict
        Dictionary of kz.fit.Peak objects to edit with the GUI
    - t_AQ: 1darray
        Acquisition timescale
    - SFO1: float
        Nucleus Larmor frequency /MHz
    - o1p: float
        Carrier frequency /ppm
    - offset: 1darray or None
        External contribution to be added to the total trace. If None, an array of zeros is used
    - I: float
        Intensity correction for the experimental spectrum
    - A0: float
        Starting total intensity for the spectrum to edit (use concentration from cal_gui)
    - rav_flag: bool
        Uses colorblind palette
    ------------
    Returns:
    - peaks: dict
        Modified kz.fit.Peak dictionary 
    - Acorr: float
        Correction for the intensity
    """

    #-----------------------------------------------------------------------
    ## USEFUL STRUCTURES
    if rav_flag:
        v_l = kz.misc.cmap2list(kz.CM['viridis'], N=10, start=0, end=1)
        colors = {'exp': v_l[2], 'act': v_l[7], 'nonact':v_l[0], 'total':'m', 'locked':v_l[4]}
    else:
        colors = {'exp': 'k', 'act':'tab:red', 'nonact':'tab:blue', 'total':'b', 'locked':'g'}

    class DummyEvent:
        """ represent a null_event """
        def __init__(self):
            self.button = None
            self.key = None

    def rename_dic(dic, Np):
        """
        Change the keys of a dictionary with a sequence of increasing numbers, starting from 1.
        ----------
        Parameters:
        - dic: dict
            Dictionary to edit
        - Np: int
            Number of peaks, i.e. the sequence goes from 1 to Np
        ----------
        Returns:
        - new_dic: dict
            Dictionary with the changed keys
        """
        old_keys = list(dic.keys())         # Get the old keys
        new_keys = [int(i+1) for i in np.arange(Np)]    # Make the new keys
        new_dic = {}        # Create an empty dictionary
        # Copy the old element in the new dictionary at the correspondant point
        for old_key, new_key in zip(old_keys, new_keys):
            new_dic[new_key] = dic[old_key]
        del dic
        return new_dic

    def calc_total(peaks, offset):
        """
        Calculate the sum trace from a collection of peaks stored in a dictionary.
        If the dictionary is empty, returns an array of zeros.
        ---------
        Parameters:
        - peaks: dict
            Components
        --------
        Returns:
        - total: 1darray
            Sum spectrum
        """
        # Get the arrays from the dictionary
        T = [p(A) for key, p in peaks.items() if key in unlocked_keys]
        if len(T) > 0:  # Check for any peaks
            total = np.sum(T, axis=0)
            return total + offset
        else:
            return offset

    #-------------------------------------------------------------------------------
    # Define the null event to handle no-interactions
    null_event = DummyEvent()

    # Remove the imaginary part from the experimental data and make a shallow copy
    if np.iscomplexobj(exp):
        S = np.copy(exp).real / I
    else:
        S = np.copy(exp) / I

    unlocked_keys = []  # Contains the keys of the unlocked peaks

    # Backup
    _peaks = deepcopy(peaks)    
    _unlocked_keys = []
    # Get initial intensity for correct computation of the intensity at the end
    r_i, I_init = kz.misc.molfrac([peak.k for _, peak in peaks.items()])
    N = S.shape[-1]     # Number of points
    Np = len(peaks.keys())              # Number of peaks

    # Make an acqus dictionary based on the input parameters.
    acqus = {'t1': t_AQ, 'SFO1': SFO1, 'o1p': o1p}

    # Set limits
    limits = [max(ppm_scale), min(ppm_scale)]
    
    # Get point indices for the limits
    lim1 = kz.misc.ppmfind(ppm_scale, limits[0])[0]
    lim2 = kz.misc.ppmfind(ppm_scale, limits[1])[0]
    # Calculate the absolute intensity (or something that resembles it)
    if A0 is None:
        A0 = np.trapz(S[lim1:lim2], dx=kz.misc.calcres(ppm_scale*SFO1))*2*kz.misc.calcres(acqus['t1'])
    _A = 1 * A0
    # This A is the one that is modified when unlocked
    A = A0
    # Make a sensitivity dictionary
    sens = {
            'u': kz.misc.calcres(ppm_scale) * 20,    # 20 pts
            'fwhm': 2.5,
            'k': 0.05,
            'b': 0.1,
            'phi': 10,
            'A': 10**(np.floor(np.log10(A)-1))    # approximately
            }
    _sens = dict(sens)                          # RESET value

    if offset is None:
        offset = np.zeros_like(S)

    # Initial figure
    fig = plt.figure('Manual Computation of Inital Guess')
    fig.set_size_inches(15,8)
    plt.subplots_adjust(bottom=0.10, top=0.90, left=0.05, right=0.65)
    ax = fig.add_subplot(1,1,1)
    
    # make boxes for widgets
    slider_box = plt.axes([0.68, 0.10, 0.01, 0.65])     # Peak selector slider
    peak_box = plt.axes([0.72, 0.45, 0.10, 0.30])       # Radiobuttons
    up_box = plt.axes([0.815, 0.825, 0.08, 0.075])      # Increase sensitivity button
    down_box = plt.axes([0.894, 0.825, 0.08, 0.075])    # Decrease sensitivity button
    save_box = plt.axes([0.7, 0.825, 0.085, 0.04])      # Save button
    reset_box = plt.axes([0.7, 0.865, 0.085, 0.04])     # Reset button
    group_box = plt.axes([0.76, 0.40, 0.06, 0.04])      # Textbox for the group selection
    plus_box = plt.axes([0.894, 0.65, 0.08, 0.075])     # Add button
    minus_box = plt.axes([0.894, 0.55, 0.08, 0.075])    # Minus button
    lock_box = plt.axes([0.894, 0.35, 0.08, 0.075])     # Lock/Unlock button
    
    # Make widgets
    #   Buttons
    up_button = Button(up_box, r'$\uparrow$', hovercolor = '0.975')    
    down_button = Button(down_box, r'$\downarrow$', hovercolor = '0.975')
    save_button = Button(save_box, 'SAVE', hovercolor = '0.975')
    reset_button = Button(reset_box, 'RESET', hovercolor = '0.975')
    plus_button = Button(plus_box, '$+$', hovercolor='0.975')
    minus_button = Button(minus_box, '$-$', hovercolor='0.975')
    lock_button = Button(lock_box, 'UNLOCK', hovercolor='0.975')

    #   Textbox
    group_tb = TextBox(group_box, 'Group', textalignment='center')
    
    #   Radiobuttons
    peak_name = [r'$\delta$ /ppm', r'$\Gamma$ /Hz', '$k$', '$x_{g}$', r'$\phi$', '$A$']
    peak_radio = RadioButtons(peak_box, peak_name, activecolor='tab:blue')      # Signal parameters
    
    #   Slider
    slider = Slider(ax=slider_box, label='Active\nSignal', valmin=0, valmax=1-1e-3, valinit=0, valstep=1/Np, orientation='vertical', color='tab:blue')


    #-------------------------------------------------------------------------------
    ## SLOTS

    def radio_changed(event):
        """ Change the printed value of sens when the radio changes """
        active = peak_name.index(peak_radio.value_selected)
        param = list(sens.keys())[active]
        write_sens(param)

    def up_sens(event):
        """ Doubles sensitivity of active parameter """
        nonlocal sens
        active = peak_name.index(peak_radio.value_selected)
        param = list(sens.keys())[active]
        sens[param] *= 2
        write_sens(param)

    def down_sens(event):
        """ Halves sensitivity of active parameter """
        nonlocal sens
        active = peak_name.index(peak_radio.value_selected)
        param = list(sens.keys())[active]
        sens[param] /= 2
        write_sens(param)

    def up_value(param, idx):
        """ Increase the value of param of idx-th peak """
        # Allow edit only if it is unlocked
        if idx not in unlocked_keys:
            return
        if param == 'A':        # It is outside the peaks dictionary!
            nonlocal A
            A += sens['A']
        else:
            nonlocal peaks
            peaks[idx].__dict__[param] += sens[param]
            # Make safety check for b
            if peaks[idx].b > 1:
                peaks[idx].b = 1

    def down_value(param, idx):
        """ Decrease the value of param of idx-th peak """
        # Allow edit only if it is unlocked
        if idx not in unlocked_keys:
            return
        if param == 'A':    # It is outside the peaks dictionary!
            nonlocal A
            A -= sens['A']
        else:
            nonlocal peaks
            peaks[idx].__dict__[param] -= sens[param]
            # Safety check for fwhm
            if peaks[idx].fwhm < 0:
                peaks[idx].fwhm = 0
            # Safety check for b
            if peaks[idx].b < 0:
                peaks[idx].b = 0

    def scroll(event):
        """ Connection to mouse scroll """
        if Np == 0: # No peaks!
            return
        # Get the active parameter and convert it into Peak's attribute
        active = peak_name.index(peak_radio.value_selected)
        param = list(sens.keys())[active]
        # Get the active peak
        idx = int(np.floor(slider.val * Np) + 1)
        # Fork for up/down
        if event.button == 'up':
            up_value(param, idx)
        if event.button == 'down':
            down_value(param, idx)

        # Recompute the components
        for k in peaks.keys():
            if k not in unlocked_keys:
                continue
            p_sgn[k].set_ydata(peaks[k](A)[lim1:lim2])
        # Recompute the total trace
        p_fit.set_ydata(calc_total(peaks, offset+locked_total)[lim1:lim2])
        # Update the text
        if idx in unlocked_keys:
            write_par(idx)
        plt.draw()

    def normalize():
        nonlocal peaks, A
        K0 = [peaks[k].k for k in unlocked_keys]
        Kn, N0 = kz.misc.molfrac(K0)
        for k, key in enumerate(unlocked_keys):
            peaks[key].k = Kn[k]
        A *= N0
        scroll(null_event)

    def write_par(idx):
        """ Write the text to keep track of your amounts """
        if idx in unlocked_keys:     # Write the things
            dic = dict(peaks[idx].par())
            dic['A'] = A
            # Update the text
            values_print.set_text('{u:+7.3f}\n{fwhm:5.3f}\n{k:5.3f}\n{b:5.3f}\n{phi:+07.3f}\n{A:5.2e}\n{group:5.0f}'.format(**dic))
            # Color the heading line of the same color of the trace
            head_print.set_color(p_sgn[idx].get_color())
        else:   # Clear the text and set the header to be black
            values_print.set_text('')
            head_print.set_color('k')
        plt.draw()

    def write_sens(param):
        """ Updates the current sensitivity value in the text """
        text = r'Sensitivity: $\pm$'+f'{sens[param]:10.4g}'
        # Update the text
        sens_print.set_text(text)
        # Redraw the figure
        plt.draw()

    def set_group(text):
        """ Set the attribute 'group' of the active signal according to the textbox """
        nonlocal peaks
        if not Np:  # Clear the textbox and do nothing more
            group_tb.text_disp.set_text('')
            plt.draw()
            return
        # Get active peak
        idx = int(np.floor(slider.val * Np) + 1)
        try:
            group = int(eval(text))
        except:
            group = peaks[idx].group
        group_tb.text_disp.set_text('')
        peaks[idx].group = group
        write_par(idx)

    def selector(event):
        """ Update the text when you move the slider """
        idx = int(np.floor(slider.val * Np) + 1)
        if Np:
            for key, line in p_sgn.items():
                if key == idx: 
                    p_sgn[key].set_lw(3)
                else:
                    p_sgn[key].set_lw(0.8)
            write_par(idx)

    def key_binding(event):
        """ Keyboard """
        key = event.key
        if key == '<':
            down_sens(0)
        if key == '>':
            up_sens(0)
        if key == '+':
            add_peak(0)
        if key == '-':
            remove_peak(0)
        if key == 'pagedown':
            if slider.val - slider.valstep >= 0:
                slider.set_val(slider.val - slider.valstep)
            selector(0)
        if key == 'pageup':
            if slider.val + slider.valstep < 1:
                slider.set_val(slider.val + slider.valstep)
            selector(0)
        if key == 'up' or key == 'down':
            event.button = key
            scroll(event)
        if key == 'N':
            normalize()

    def reset(event):
        """ Return everything to default """
        nonlocal Np, peaks, p_sgn, A, sens
        # Lock everything
        lock()
        # Reset all the rest
        A = _A
        sens = dict(_sens)
        ax.set_xlim(*_xlim)
        ax.set_ylim(*_ylim)

        # Re-initialize everything
        peaks = deepcopy(_peaks)
        Np = len(peaks.keys())
        slider.valstep = 1/Np
        # Re-lock because otherwise it does not compute total_locked correctly
        lock()
        # Draw the stuff
        scroll(null_event)
        # Reset the label on the LOCK/UNLOCK button to default
        lock_button.label.set_text('UNLOCK')
        plt.draw()

    def add_peak(event):
        """ Add a component """
        nonlocal Np, peaks, p_sgn
        # Increase the number of peaks
        Np += 1 
        # Add an entry to the dictionary labelled as last
        peaks[Np] = kz.fit.Peak(acqus, u=np.mean(ax.get_xlim()), N=N, group=0)
        # Plot it and add the trace to the plot dictionary
        p_sgn[Np] = ax.plot(ppm_scale[lim1:lim2], peaks[Np](A)[lim1:lim2], lw=0.8)[-1]
        # Move the slider to the position of the new peak
        slider.set_val( (Np - 1) / Np )
        # Recompute the step of the slider
        slider.valstep = 1 / Np
        # Calculate the total trace with the new peak
        total = calc_total(peaks, offset)
        # Update the total trace plot
        p_fit.set_ydata(total[lim1:lim2])
        # Update the text
        write_par(Np)

    def remove_peak(event):
        """ Remove the active component """
        nonlocal Np, peaks, p_sgn
        if Np == 0:
            return
        # Get the active peak
        idx = int(np.floor(slider.val * Np) + 1)
        # Allow edit only on unlocked peaks
        if idx not in unlocked_keys:
            return
        # Decrease Np of 1
        Np -= 1
        # Delete the entry from the peaks dictionary
        _ = peaks.pop(idx)
        # Remove the correspondant line from the plot dictionary
        del_p = p_sgn.pop(idx)
        # Set it invisible because I cannot truly delete it
        del_p.set_visible(False)
        del del_p   # ...at least clear some memory
        # Change the labels to the dictionary
        peaks = rename_dic(peaks, Np)
        p_sgn = rename_dic(p_sgn, Np)
        # Calculate the total trace without that peak
        total = calc_total(peaks, offset)
        # Update the total trace plot
        p_fit.set_ydata(total[lim1:lim2])
        # Change the slider position
        if Np == 0: # to zero and do not make it move
            slider.set_val(0)
            slider.valstep = 1e-10
            write_par(0)
        elif Np == 1:   # To zero and that's it
            slider.set_val(0)
            slider.valstep = 1 / Np
            write_par(1)
        else:   # To the previous point
            if idx == 1:
                slider.set_val(0)
            else:
                slider.set_val( (idx - 2) / Np)     # (idx - 1) -1
            slider.valstep = 1 / Np
            write_par(int(np.floor(slider.val * Np) + 1))

    def save(event):
        # Close the figure panel
        plt.close()

    def lock():
        """ Set the spectrum in LOCK mode """
        nonlocal A, unlocked_keys, locked_total, p_sgn, original_colors, peaks
        # Remove all the interactive peaks
        for key in unlocked_keys:
            item = p_sgn.pop(key)
            item.set_visible(False)
            del item
        # Correct the intensities to adhere them with A0
        for k, key in enumerate(unlocked_keys):
            peaks[key].k *= A / A0
        # Reset A to A0
        A = A0
        # Lock all the peaks by emptying unlocked_keys
        unlocked_keys = deepcopy(_unlocked_keys)
        # Compute the trace of the locked peaks
        locked_total = np.sum([peak(A) for key, peak in peaks.items() if key not in unlocked_keys], axis=0)
        # Update it
        locked_trace.set_ydata(locked_total[lim1:lim2])
        # Draw the stuff
        scroll(null_event)
        # Update the slider
        slider.set_val(0)
        slider.valstep = 1e-10
        write_par(0)

    def unlock():
        """ Set the spectrum in UNLOCK mode """
        nonlocal A, unlocked_keys, locked_total, p_sgn, original_colors
        # Get the current limits of the spectrum
        window = ax.get_xlim()
        A = A0      # Set current A to A0
        # Add the peaks inside the window as editable
        for key, peak in peaks.items():
            if min(window) <= peak.u and peak.u <= max(window):
                unlocked_keys.append(key)
        # Draw only the unlocked peaks
        for key, peak in peaks.items():
            if key not in unlocked_keys:
                continue
            p_sgn[key] = ax.plot(ppm_scale[lim1:lim2], peak(A)[lim1:lim2], lw=0.8)[-1]
            original_colors[key] = p_sgn[key].get_color()
        # Compute the locked trace with only the peaks left inactive
        if len(unlocked_keys) - Np == 0: 
            locked_total = np.zeros_like(ppm_scale)
        else:
            locked_total = np.sum([peak(A) for key, peak in peaks.items() if key not in unlocked_keys], axis=0)
        # Draw stuff
        locked_trace.set_ydata(locked_total[lim1:lim2])
        scroll(null_event)
        # Update the slider
        idx = min(unlocked_keys)
        slider.set_val( (idx - 1) / Np)     # (idx - 1) -1
        slider.valstep = 1 / Np
        write_par(idx)
        
    def lock_unlock(event):
        """ Fork for LOCK/UNLOCK button """
        if lock_button.label.get_text() == 'LOCK':
            # It means that the spectrum is in "unlock" state
            lock()
            # Change the label to "Unlock"
            lock_button.label.set_text('UNLOCK')
        elif lock_button.label.get_text() == 'UNLOCK':
            # It means that the spectrum is in "lock" state
            unlock()
            # Change the label to "Lock"
            lock_button.label.set_text('LOCK')
        plt.draw()

    #-------------------------------------------------------------------------------
    # PLOT
    ax.plot(ppm_scale[lim1:lim2], S[lim1:lim2], label='Experimental', lw=1.0, c=colors['exp'])  # experimental
    p_fit = ax.plot(ppm_scale[lim1:lim2], np.zeros_like(S)[lim1:lim2], label='Fit', lw=0.9, c=colors['total'])[-1]  # Total trace
    p_sgn = {}  # Placeholder for the plot of the components
    original_colors = {}    # For reset, save the color of the traces

    # Trace for the locked peaks
    locked_total = np.sum([peak(A) for key, peak in peaks.items() if key not in unlocked_keys], axis=0)
    locked_trace = ax.plot(ppm_scale[lim1:lim2], locked_total[lim1:lim2], lw=0.9, c=colors['locked'])[-1]
    idx = 0
    
    # Header for current values print
    head_print = ax.text(0.75, 0.35, 
            '{:>7s}\n{:>5}\n{:>5}\n{:>5}\n{:>7}\n{:>7}\n{:>7}'.format(
                r'$\delta$', r'$\Gamma$', r'$k$', r'$\beta$', 'Phase', r'$A$', 'Group'),
            ha='right', va='top', transform=fig.transFigure, fontsize=14, linespacing=1.5)
    # Text placeholder for the values - linspacing is different to align with the header
    values_print = ax.text(0.85, 0.35, '',
            ha='right', va='top', transform=fig.transFigure, fontsize=14, linespacing=1.55)
    # Text to display the active sensitivity values
    sens_print = ax.text(0.875, 0.775, f'Sensitivity: $\\pm${sens["u"]:10.4g}',
            ha='center', va='bottom', transform=fig.transFigure, fontsize=14)
    # Text to remind keyboard shortcuts
    t_uparrow = r'$\uparrow$'
    t_downarrow = r'$\downarrow$'
    keyboard_text = '\n'.join([
        f'{"KEYBOARD SHORTCUTS":^50s}',
        f'{"Key":>5s}: Action',
        f'-'*50,
        f'{"<":>5s}: Decrease sens.',
        f'{">":>5s}: Increase sens.',
        f'{"+":>5s}: Add component',
        f'{"-":>5s}: Remove component',
        f'{"Pg"+t_uparrow:>5s}: Change component, up',
        f'{"Pg"+t_downarrow:>5s}: Change component, down',
        f'{t_uparrow:>5s}: Increase value',
        f'{t_downarrow:>5s}: Decrease value',
        f'-'*50,
        ])
    keyboard_print = ax.text(0.86, 0.025, keyboard_text, 
            ha='left', va='bottom', transform=fig.transFigure, fontsize=8, linespacing=1.55)

    # make pretty scales
    ax.set_xlim(max(limits[0],limits[1]),min(limits[0],limits[1]))
    kz.misc.pretty_scale(ax, ax.get_xlim(), axis='x', n_major_ticks=10)
    kz.misc.pretty_scale(ax, ax.get_ylim(), axis='y', n_major_ticks=10)
    kz.misc.mathformat(ax)

    # Print instructions in the title
    ax.set_title('Zoom on the interested area, then UNLOCK it. Edit the peaks using the mouse scroll.\nFinally, LOCK again. Repeat. Press "SAVE" to save the results.')

    # RESET values for xlim and ylim
    _xlim = ax.get_xlim()
    _ylim = ax.get_ylim()

    # Connect the widgets to their slots
    plus_button.on_clicked(add_peak)
    minus_button.on_clicked(remove_peak)
    up_button.on_clicked(up_sens)
    down_button.on_clicked(down_sens)
    slider.on_changed(selector)
    group_tb.on_submit(set_group)
    reset_button.on_clicked(reset)
    save_button.on_clicked(save)
    lock_button.on_clicked(lock_unlock)
    peak_radio.on_clicked(radio_changed)
    fig.canvas.mpl_connect('scroll_event', scroll)
    fig.canvas.mpl_connect('key_press_event', key_binding)

    # Initialization
    scroll(null_event)

    plt.show()  # Start event loop
    plt.close()

    # Correct the intensities to preserve the integral
    r_i, I_corr = kz.misc.molfrac([peak.k for _, peak in peaks.items()])
    for key, peak in peaks.items():
        peak.k = r_i[key-1]
    Acorr = (I_corr / I_init) * (A / _A)

    return peaks, Acorr


