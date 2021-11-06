'''
###############################################################################
################## Program slot postion calculation in Python #################
########################## Niall Boohan 2020 ##################################
###################### niall.boohan@tyndall.ie ################################
###############################################################################

BASED ON:
Theory and equations sourced from:
Title: Theory of improved spectral purity in index patterned Fabry-Perot lasers
Author: O'Brien, S.; O'Reilly, E. P.
Year: 2005

Title: Spectral manipulation in Fabry-Perot lasers: perturbative inverse
Author: O'Brien, Stephen; Amann, Andreas; Fehse, Robin; Osborne, Simon;
        O'Reilly, Eoin P.; Rondinelli, James M.
Year: 2006
'''

##############################################################################
# Dashboard
##############################################################################
# Call libraries ###
import numpy as np
import matplotlib.pyplot as plt
import itertools as it

from fileinput import CL, cl_m, IND_RID, ind_def, IND_DEL, lam0_m, lam0_um,\
        AL_CAV, sw, sw_um, r_l, r_r, NO_SLOT, cl_cm, m0, ALIGN

# Import functions
from fileinput import n_group

# Code Settings  ###
POS = 2                 # Slots 0=RHS,1=LHS,2=RHS&LHS,4=Experimental for CALC=3
CALC = 2                # Function change
TC = 0                  # Turn on tolerance calc
ERR = 2                 # Turn on cavity length error of 1%
DENS = 0                # Turn on dens func plot
MIN = 1                 # Option add grat min to comp exponent
MAX = 0                 # Add grating stop not at facet
PHASE = 0               # Turn on out of phase calc
SECOND_ORD = 0          # Add 2nd order effects

# Set min/max grating manually ###
if MAX == 1:
    print('MAX IS ON!!!')
    E_max = 0.5-50/CL
if MAX == 0:
    E_max = 0.5
if CALC == 0:
    E_min = 0.0125
if CALC != 0 and MIN == 0:       # Experimental value
    E_min = 0
if CALC != 0 and MIN == 1:
    E_min = 0.0125
if MIN == 1:
    print('MIN IS ON!!!')
# Transfer values ###
al_m = AL_CAV*100       # Loss (m^-1)

# Internal values ###
MODE = 50              # +/- No of modes to consider & plot
MODE_CALC = 20          # No of modes to run calc across
MODE_NO = 20            # Modes of approximation
TAU = 0.036             # Gain modulation envelope
ALPHA = 0.627633        # Modulation to 2nd peak approx
D_INT = 20              # Set integer for slot spacing
pi = np.pi

# Converts ###
if ERR == 1:
    error = 1.001
    lam0_m = lam0_m*error              # gain_factdd 1% err wl
    cl_m = cl_m*error                  # gain_factdd 1% err cav length
    CL = CL*error

# Store outputs ###
B_cav = []
mo = []


###############################################################################
# Weighting functions ###
###############################################################################
def weight_func_0(ep):
    return 1/f_ep(ep, gam0_cm, cl_cm)


def weight_func_1(ep):
    return ep/f_ep(ep, gam0_cm, cl_cm)


def weight_func_3(ep):
    return (np.sin((3/2)*pi*ep)**2)/f_ep(ep, gam0_cm, cl_cm)


def weight_func_4(ep):
    return (np.sin(pi*ep)**2)/f_ep(ep, gam0_cm, cl_cm)


def weight_func_5(ep):
    return ep*np.sin(2*pi*ep)/f_ep(ep, gam0_cm, cl_cm)


###############################################################################
# Main body functions ###
###############################################################################
# Generates array through zero evenly spaced ###
def mirrored(maxval, inc=1):
    x = np.arange(inc, maxval, inc)
    if x[-1] != maxval:
        x = np.r_[x, maxval]
    return np.r_[-x[::-1], 0, x]


# Finds 2nd smallest value in list ###
def second_smallest(numbers):
    m1 = m2 = float('inf')
    for x in numbers:
        if x <= m1:
            m1, m2 = x, m1
        elif x < m2:
            m2 = x
    return m2


# Facet modulation ###
def f_ep(ep, g, c):
    return r_l*np.exp(ep*g*c) - r_r*np.exp(-ep*g*c)


# Gamma calculation for modal approximation ###
def Gam(ep, n):
    return np.exp(-np.pi*(ep-n/MODE_NO)**2/TAU**2)


# Finds the nearest value in array to a number ###
def closest(lst, K):
    lst = np.asarray(lst)
    idx = (np.abs(lst - K)).argmin()
    return idx, lst[idx]


# Peak cavity functions from having odd number of QW length from left facet ###
def peak_cav_right(sp_um):
    gap = lambda s: (CL - s)/(lam0_um/(IND_RID*4))
    rnd = round((gap(sp_um)-1)/2)*2+1
    if gap(sp_um) - rnd < 0:
        while gap(sp_um) - rnd < 0:
            sp_um -= 1e-7
    if gap(sp_um) - rnd > 0:
        while gap(sp_um) - rnd > 0:
            sp_um += 1e-7
    return sp_um


def peak_cav_left(sp_um):
    gap = lambda s: s/(lam0_um/(IND_RID*2))
    rnd = round(gap(sp_um))
    if gap(sp_um) - rnd < 0:
        while gap(sp_um) - rnd < 0:
            sp_um += 1e-7
    if gap(sp_um) - rnd > 0:
        while gap(sp_um) - rnd > 0:
             sp_um -= 1e-7
    return sp_um


def peak_ep_cos(ep):
    diff = 1e-8
    step = 1e-8
    sign = 1
    if m0 % 2 == 0:
        sign = 1
    if m0 % 2 != 0:
        sign = -1
    if ep > mid_point:
        if np.cos(2*(ep+diff)*pi*m0) < np.cos(2*ep*pi*m0):
            while abs(np.cos(2*ep*pi*m0)-1*sign) > step:
                ep -= diff
        if np.cos(2*(ep+diff)*pi*m0) > np.cos(2*ep*pi*m0):
            while abs(np.cos(2*ep*pi*m0)-1*sign) > step:
                ep += diff
    if ep < mid_point:
        if np.cos(2*(ep+diff)*pi*m0) < np.cos(2*ep*pi*m0):
            while abs(np.cos(2*ep*pi*m0)+1*sign) > step:
                ep += diff
        if np.cos(2*(ep+diff)*pi*m0) > np.cos(2*ep*pi*m0):
            while abs(np.cos(2*ep*pi*m0)+1*sign) > step:
                ep -= diff
    return ep


def peak_ep_sin(ep):
    diff = 1e-8
    step = 1e-9
    if m0 % 2 == 0:
        sign = 1
    if m0 % 2 != 0:
        sign = -1
    if ep > mid_point:
        if np.sin(2*(ep+diff)*pi*m0) < np.sin(2*ep*pi*m0):
            while abs(np.sin(2*ep*pi*m0)-1*sign) > step:
                ep -= diff
        if np.sin(2*(ep+diff)*pi*m0) > np.sin(2*ep*pi*m0):
            while abs(np.sin(2*ep*pi*m0)-1*sign) > step:
                ep += diff
    if ep < mid_point:
        if np.sin(2*(ep+diff)*pi*m0) < np.sin(2*ep*pi*m0):
            while abs(np.sin(2*ep*pi*m0)+1*sign) > step:
                ep += diff
        if np.sin(2*(ep+diff)*pi*m0) > np.sin(2*ep*pi*m0):
            while abs(np.sin(2*ep*pi*m0)+1*sign) > step:
                ep -= diff
    return ep


# SMSR calc from Mike Wallace paper ###
def SMSR(threshold_gain, Delta_mode):
    gain_diff = [i - min(threshold_gain) for i in threshold_gain]
    del_gain = 1e-3*(15/(100-15))
    gain_band = min(threshold_gain)-(2*(Delta_mode/20))**2
    gain_band = [max(gain_band)-i for i in gain_band]
    smsr = [10*np.log10((i+j)/del_gain+1) for i, j in
            zip(gain_band, gain_diff)]
    return smsr


# Returns gain in cm^-1 and mode WL nm ###
def gain_thresh(ep, m, plot):

    # Initial calculations ###
    mode_lst = mirrored(m, 1)
    mode_cav = [mode_lst[i]+m0 for i in range(len(mode_lst))]
    mode_wl_um = [2*CL*IND_RID/mode_cav[i] for i in range(len(mode_lst))]
    mode_wl_nm = [mode_wl_um[i]*1e3 for i in range(len(mode_lst))]
    k_um = [2*np.pi/mode_wl_um[i] for i in range(len(mode_lst))]
    theta_j = [k_um[i]*sw_um*ind_def for i in range(len(mode_lst))]
    theta_k = [k_um[i]*sw_um*ind_def for i in range(len(mode_lst))]
    mod_fact_one = [[-1*np.sin(theta_j[j])*np.cos(m0*pi)*np.cos(mode_lst[j]*pi)
                     * (np.sin(2*pi*ep[i]*m0)*np.cos(2*pi*ep[i]*mode_lst[j])
                        + np.cos(2*pi*ep[i]*m0)*np.sin(2*pi*ep[i]*mode_lst[j]))
                     * f_ep(ep[i], gam0_cm, cl_cm) for i in range(len(ep))]
                    for j in range(len(mode_lst))]
    mod_fact_one = [sum(i) for i in mod_fact_one]
    mod_fact_one_ep = [[ep[i]*np.sin(theta_j[j])*np.cos(m0*pi)
                        * np.cos(mode_lst[j]*pi)*(np.sin(2*pi*ep[i]*m0)
                        * np.cos(2*pi*ep[i]*mode_lst[j])+np.cos(2*pi*ep[i]*m0)
                        * np.sin(2*pi*ep[i]*mode_lst[j]))
                        * f_ep(ep[i], gam0_cm, cl_cm) for i in range(len(ep))]
                       for j in range(len(mode_lst))]
    mod_fact_one_ep = [sum(i) for i in mod_fact_one_ep]
    mod_fact_two = [[np.sin(theta_j[j])*np.sin(theta_k[j])
                     * np.sinh((ep[i+1]-ep[i])*cl_cm*gam0_cm)
                     * np.sin(2*ep[i]*m0*pi)*np.sin(2*ep[i+1]*m0*pi)
                     * np.cos(2*(ep[i+1]-ep[i])*mode_lst[j]*pi)
                     for i in range(len(ep)-1)] for j in range(len(mode_lst))]
    mod_fact_two = [sum(i) for i in mod_fact_two]

    # Calculate 1st order mode shift ###
    delta_mode_1 = [[np.cos(m0*pi)*np.cos(mode_lst[j]*pi)*np.sign(ep[i])
                     * np.sin(2*ep[i]*mode_lst[j]*pi) for i in range(len(ep))]
                    for j in range(len(mode_lst))]
    delta_mode_1 = [sum(i) for i in delta_mode_1]

    # Gain totals calculations ###
    g_1 = [fact*mod_fact_one[i] for i in range(len(mode_lst))]
    g_2 = [(fr_n**2)*g_1[i]*(1/(np.sqrt(r_l*r_r)))*mod_fact_one_ep[i]
           + (2/cl_cm)*(delta_mode_1[i]**2+mod_fact_two[i])
           for i in range(len(mode_lst))]
    if SECOND_ORD == 0:
        g_mode = [gam0_cm + fr_n*g_1[i] for i in range(len(mode_lst))]
    if SECOND_ORD == 1:
        g_mode = [gam0_cm + fr_n*g_1[i] + (fr_n**2)*g_2[i] for i in
                  range(len(mode_lst))]

    # Calculate totals ###
    delta_mode = [fr_n*delta_mode_1[i] for i in range(len(delta_mode_1))]
    mode_shft = [mode_cav[i] + delta_mode[i] for i in range(len(mode_lst))]
    mode_wl_m_shft = [(2*cl_m*IND_RID)/mode_shft[i]
                      for i in range(len(mode_lst))]
    mode_wl_nm_shft = [mode_wl_m_shft[i]*1e9 for i in range(len(mode_lst))]
    delta_mode_wl = [1e3*(mode_wl_nm[i] - mode_wl_nm_shft[i]) for i in
                     range(len(mode_shft))]

    if plot == 1:
        plt.plot(mode_lst, [fr_n*i for i in g_1],
                 'g:*', label='1$^{st}$ order')
        plt.title('1$^{st}$ Order')
        plt.ylabel('$\Delta$ $G_{th}$')
        plt.xlabel('m')
        plt.show()
        if SECOND_ORD == 1:
            plt.plot(mode_lst, [(fr_n**2)*i for i in g_2],
                     '-*', label='2$^{nd}$ order')
            plt.title('2$^{nd}$ Order')
            plt.ylabel('$\Delta$ $G_{th}$')
            plt.xlabel('m')
            plt.show()

    # Supression calculation ###
    sort_g = sorted(g_mode)
    g_wl = mode_wl_nm_shft[g_mode.index(sort_g[0])]
    sup_b = sort_g[1]-sort_g[0]
    
    #if g_mode[g_mode.index(sort_g[0])] == 0:
    #    sup_nn = g_mode[g_mode.index(sort_g[0])+1] - sort_g[0]
    #    break;
    #if g_mode[g_mode.index(sort_g[0])+1] < g_mode[g_mode.index(sort_g[0])-1]:
    #   sup_nn = g_mode[g_mode.index(sort_g[0])+1] - sort_g[0]
    #else:
    #    sup_nn = g_mode[g_mode.index(sort_g[0])-1] - sort_g[0]
    sup_nn = 0
    print('Cavity wavelength with grating = ', g_wl, 'nm')
    print('Mode suppression across band = ', sup_b, 'cm^1')
    print('Mode suppression nearest neighbour = ', sup_nn, 'cm^1')
    print('Gain min = ', sort_g[0], 'cm^-1')
    return g_mode, mode_lst, mode_wl_nm,  mode_wl_nm_shft, delta_mode_wl,\
        delta_mode, sup_b, g_1


# Cal density functions ###
def func(mode_num, ep, split):
    mm = [cav_space.index(j)+1 for j in ep]             # Convert to mode space
    k = [-ALPHA*((mm[j]/m0)-0.5) for j in range(len(ep))]
    # Kevin & Stephen method ###
    if mode_num == 0:
        b = [[(Gam(ep[j], N[i])/f_ep(ep[j], gam0_m, cl_m))
              for i in range(len(N))] for j in range(len(ep))]
        b = [sum(i) for i in b]
    # Masood method ###
    if mode_num == 1:
        b = [np.exp(k[j]) for j in range(len(ep))]
    # Additional weighting ###
    if mode_num == 2:
        if split == 0:
            #############################################################
            #############################################################
            #############################################################
            b = [weight_func_2(ep[j]) for j in range(len(ep))]
            #############################################################
            #############################################################
            #############################################################
        if split == 1:
            b = [weight_func_0(ep[j]) for j in range(len(ep))]
    return b


# Function for setting constant spacing design ###
def set_d(integ):
    d = integ*lam0_um/IND_RID
    FSR = (lam0_um**2)/(n_group(lam0_um, 0, IND_RID)*d)
    d_ep = d/CL
    ep = [d_ep+i*d_ep for i in range(NO_SLOT)]
    print('FSR = ', FSR*1e3, 'nm')
    print('d =', d, 'um')
    return ep


###############################################################################
# Plotting definitions ###
###############################################################################
# Gain plot ###
def gain_plt(wl, g, val):
    fig, ax = plt.subplots()
    ax.plot(wl, g, 'o--')
    if val == 0:
        ax.set_title('Threshold gain fourier')
    if val != 0:
        ax.set_title('$\Delta$ = %f $\mu m$' %val)
    ax.set_xlabel('$\lambda$ (nm)')
    ax.set_ylabel('$\Delta$G$_{th}$ (cm$^{-1}$)')
    ax.hlines(gam0_cm, wl[0], wl[-1], colors='r')
    plt.show()
    return


def smsr_plt(m, smsr):
    plt.plot(m, [i*-1 for i in smsr], ':x')
    plt.title('SMSR')
    plt.xlabel('$\lambda$ (nm)')
    plt.ylabel('SMSR (dB)')
    plt.show()
    return


# Slot postion  plot ###
def slot_plot(ep):
    fig, ax = plt.subplots()
    plt.bar(ep, height=1, width=1e-3)
    ax.set_title('Position')
    ax.set_xlim(-1/2, 1/2)
    ax.set_xlabel('Displacement ($\epsilon$)')
    ax.set_ylim(0, 1)
    ax.get_yaxis().set_visible(False)
    plt.show()
    return


def plot_delta_mode(m, d):
    plt.plot(m, d, '-o')
    plt.hlines(0, m[0], m[-1], colors='r')
    plt.xlabel('Mode')
    plt.ylabel('Mode shift ($\Delta$m)')
    plt.show()
    return


###############################################################################
# Calculation definition ###
###############################################################################
def calc_func(ep, split, no):
    # Reset lists ###
    Lf = []
    L = []

    # Function calculation ###
    B = func(CALC, ep, split)

    # Feature density ###
    if DENS == 1:
        plt.plot(ep, B)
        plt.title('Feature Density')
        plt.xlabel('Disp $(\epsilon)$')
        plt.ylabel('A.U.')
        plt.show()

    # Calc absolute func value ###
    B = [abs(i) for i in B]
    B_cav.append(B)                     # B function for entire cavity length
    B = np.where(np.isnan(B), 0, B)     # Remove and NaNs
    C = list(it.accumulate(B))          # Integrate
    ws = max(C)                         # Weighted sum, max val of C

    # Find positions in C as proportion of max C
    l_idx = [C-(ws*(i-0.5)/no) for i in range(no+1)]

    # Sort these postion to coincide with fractions of positions in C ###
    for i in range(no):
        k, l = 0, 0
        for j in l_idx[i+1]:
            if j > 0:
                l = k
            k += 1
            Lf.append(l)

    # Sort the first point in grating space as slot position ###
    for i in range(len(Lf)-1):
        if Lf[i-1] == 0 and Lf[i+1] > 0 and Lf[i] != 0:
            L.append(Lf[i])
        pos = [ep[i] for i in L]         # Converts index to cav position

    ep = []
    return pos


###############################################################################
# Pre Calculations ###
###############################################################################
gam0_m = al_m-(1/(2*cl_m))*np.log((r_l**2)*(r_r**2))
gam0_cm = gam0_m/1e2                       # Cavity gain (cm^-1)
gain_fact = gam0_cm*cl_cm
beta = sw/cl_m
fr_n = IND_DEL/IND_RID
fact = 1/(cl_cm*np.sqrt(r_l*r_r))
print('Cavity threshold gain =', gam0_cm, 'cm^-1')
print('Gain factor =', gain_fact)

# Calc eps table positions ###
mid_point = -np.log(r_l/r_r)/(2*gain_fact)           # mid point cavity
print('Cavity mid point = ', mid_point, 'eps')
cav_space = np.linspace(-1/2, 1/2, 10000)
cav_space = [i for i in cav_space]
N = mirrored(MODE_CALC, 1)                           # No modes to sweep
print('m0 = ', m0)
print('Bare cavity wavelength = ', 2*CL*1e3*IND_RID/m0, 'nm')
print('sw = ', sw_um, 'um')

# Define point along cavity to calculate ###
E_r = [i for i in cav_space if i > E_min and i < E_max]
E_l = [i for i in cav_space if i < - E_min]
E_exp1 = [i for i in cav_space if i > 0.0 and i < 0.25]
E_exp2 = [i for i in cav_space if i > 0.25 and i < 0.5]


###############################################################################
# Main ###
###############################################################################
if POS == 0 and CALC != 3:
    eps = calc_func(E_r, 0, NO_SLOT)
if POS == 1 and CALC != 3:
    eps = calc_func(E_l, 0, NO_SLOT)
if POS == 2 and CALC != 3:
    eps = calc_func(E_l, 0, NO_SLOT) + calc_func(E_r, 0, NO_SLOT)
    NO_SLOT = len(eps)
if POS == 4 and CALC != 3:
    exp = [i for i in cav_space if i > 0]
    eps = calc_func(exp, 0, NO_SLOT)
if POS == 5 and CALC != 3:
    eps = calc_func(E_r, 0, NO_SLOT)
if POS == 0 and CALC == 3:
    eps = set_d(D_INT)
if POS == 6 and CALC == 2:
    eps = calc_func(E_exp1, 0, 10)+calc_func(E_exp2, 1, 10)

# Index position calculation, overwritten if sin peak is on ###
if CALC == 1:
    len_ind = [closest(cav_space, i) for i in eps]
    len_ind = [len_ind[i][0] for i in range(len(len_ind))]

# Jack Align
if ALIGN == 0:
    eps = [eps[i]+(i+1/2)*beta*fr_n for i in range(len(eps))]
    pos_slot = [(eps[i]+1/2)*CL for i in range(len(eps))]
    pos_slot = [pos_slot[i] + sw_um/2 for i in range(len(eps))]
    pos_slot = [peak_cav_right(pos_slot[i]) for i in range(len(eps))]
    eps = [pos_slot[i]/CL - 1/2 for i in range(len(eps))]
    eps = [eps[i] - (sw_um)/CL for i in range(len(eps))]
    eps = [peak_ep_cos(eps[i]) for i in range(len(eps))]
    pos_slot = [(eps[i]+1/2)*CL for i in range(len(eps))]
    eps_four = [eps[i] + (sw_um/2)/CL for i in range(len(eps))]

if ALIGN == 1:
    eps = [eps[i]+(i+1/2)*beta*fr_n for i in range(len(eps))]
    pos_slot = [(eps[i]+1/2)*CL for i in range(len(eps))]
    pos_slot = [pos_slot[i] + sw_um/2 for i in range(len(eps))]
    pos_slot = [peak_cav_left(pos_slot[i]) for i in range(len(eps))]
    eps = [pos_slot[i]/CL - 1/2 for i in range(len(eps))]
    eps = [eps[i] + (sw_um/2)/CL for i in range(len(eps))]
    eps = [peak_ep_sin(eps[i]) for i in range(len(eps))]
    pos_slot = [(eps[i]+1/2)*CL for i in range(len(eps))]
    eps_four = [eps[i] for i in range(len(eps))]


# Introduce arbitrary error ###
if ERR == 1:
    eps_four = [eps_four[i]/error for i in range(len(eps))]
if ERR == 2:
    eps_four = [eps_four[i] + 20/CL + 0.5*(1.55/(16*3.2))/CL for i in range(len(eps))]
    #eps_four = [eps_four[i] + 20/CL + 0.156/CL for i in range(len(eps))]
    pos_slot = [(eps[i]+1/2)*CL for i in range(len(eps))]
    pos_slot = [(eps_four[i]+1/2)*CL for i in range(len(eps))]
    #pos_slot = [i - 20 for i in pos_slot]


#if len(pos_slot) == 40:
    #pos_slot.pop(0)

print('Cavity length = ', CL, 'um')

# Run gain calc func ###
gain, mode_hold, mode_wl_nm, mode_shft_wl,\
    delta_mode_wl, mode_shft, sup_b, del_gain = gain_thresh(eps_four, MODE, 1)
print(gain)
print(mode_wl_nm)
print(pos_slot)
smsr = SMSR(gain, mode_hold)
###############################################################################
# Visualisation & Output ###
###############################################################################
# plot ###
gain_plt(mode_wl_nm, gain, 0)
plot_delta_mode(mode_hold, mode_shft)
slot_plot(eps)
smsr_plt(mode_hold, smsr)
# Save, mode shift not included  ###
np.savez('InvGen.npz', pos_slot=pos_slot, WL=mode_wl_nm, gain=gain, sw=sw_um)


###############################################################################
# Modal impact of each slot ###
###############################################################################
sumode_sinh = 2*fr_n*r_r*sum([np.sinh(eps[j]*np.log(1/(r_r*r_l)))
                              for j in range(len(eps))])
print('Sum slot effect =', sumode_sinh/cl_cm)


###############################################################################
# Tolerance calc ###
###############################################################################
def tol_calc():

    PLOT_ALL = 1    # Subroutine to plot all spectra

    # Generate lists to hold output values ###
    s, d, g_list, sp, smsr_lst = ([] for i in range(5))
    len_ind = [(eps_four[i]+1/2)*m0 for i in range(len(eps))]
    trim = len_ind[:len(len_ind)-0]     # Fnc to trim values

    # Calc error in wl and conver into mode space ###
    N = 5
    L = (lam0_m/(2*IND_RID))/N
    #L = 2.5*(lam0_m/(IND_RID))/N
    step = L/cl_m
    print('Step = ', step)
    for i in range(0, N+1):
        G_tol = []
        dh = (i*step)*CL      # Output step in um
        for j in range(len(trim)):
            calc = (trim[j]/m0-1/2)+i*step
            G_tol.append(calc)
        g, mh, sh, NULL, NULL, NULL, sup, NULL = gain_thresh(G_tol, MODE, 0)
        g_list.append(g)
        d.append(dh)
        s.append(sh)
        mo.append(mh)
        sp.append(sup)
        smsr = SMSR(g, mh)
        print('SMSR = ', second_smallest(smsr), 'dB') 
        smsr_lst.append(second_smallest(smsr))
    print(dh)
    print(sp)

    if PLOT_ALL == 1:
        for i in range(len(g_list)):
            plt.plot(mh, g_list[i], label='%3f $\mu m$' % d[i])
        plt.legend()
        plt.xlabel('Mode')
        plt.ylabel('Threshold Gain $(cm^{-1})$')
        plt.show()

    plt.plot(d, sp, '--')
    plt.xlabel('Distance $(\mu m)$')
    plt.ylabel('Min Modal Discrimination $(\Delta cm^{-1})$')
    plt.show()

    plt.plot(d, smsr_lst, '--')
    plt.xlabel('Distance $(\mu m)$')
    plt.ylabel('Min SMSR (dB)')
    plt.show()

    print(d)
    print(smsr_lst)


def plot_QW():
    len_ind = [(eps_four[i]+1/2)*m0 for i in range(len(eps))]
    trim = len_ind[:len(len_ind)-0]     # Fnc to trim values
    # Calc error in wl and conver into mode space ###
    L = lam0_m/(8*IND_RID)
    #L = lam0_m/(16*IND_RID)
    step = L/cl_m
    #step = 0
    #print('Step = ', L*1e6,'um')
    G_tol = []
    for j in range(len(trim)):
        calc = (trim[j]/m0-1/2)+step
        G_tol.append(calc)
    g, mh, m_wl, NULL, NULL, NULL, sup = gain_thresh(G_tol, MODE, 1)
    #plt.plot(mh, [i*-1+gam0_cm for i in g], ':*')
    s = SMSR(g, mh)
    print(s)
    plt.plot(mh, g, ':*')
    plt.title('Flipped to Loss')
    plt.xlabel('Mode')
    plt.ylabel('Threshold Loss $(cm^{-1})$')
    plt.show()
    print(g)
    print(m_wl)


if PHASE == 1:
    plot_QW()
if TC == 1:
    tol_calc()
