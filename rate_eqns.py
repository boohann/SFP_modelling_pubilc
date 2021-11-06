'''
###############################################################################
########### Program to simulate laser-rate equations in Python ################
########################## Niall Boohan 2020 ##################################
###################### niall.boohan@tyndall.ie ################################
###############################################################################

BASED ON:
Theory and equations sourced from:
Title: Diode Lasers and Photonic Integrated Circuits
Author: Coldren, Larry A.
Year: 1997
Compensation values from:
Title: Comparison of rate-equation and Fabry-Perot approaches to modeling a
       diode laser
Author: Daniel T. Cassidy
Year: 1983
-------------------------------------------------------------------------------
NOTES:
-Multimode calculation based on https://en.wikipedia.org/wiki/
 Laser_diode_rate_equations
-Radiative and non-radiative recombination Schubert, E. Fred
'''

# Import necessary libraries ###
from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
import functools as fn

# Import parameters ###
from fileinput import CUR, cur_sweep_mA, AL_CAV, LAM0, GAM, A, B, C, cl_cm,\
    DG, EPS, M, q, vol_cm, vol_nm, vg_cm, f_Hz, r_l, r_r, IND_RID, D_IND, c, al_t, R_SP

# Import functions ###
from fileinput import p_mW, p_dBm, tp_m

###############################################################################
# Dashboard
###############################################################################
# Settings ###
MM = 1              # 0 single mode, 1 multiple modes
LI = 0              # Turn on LI sweep
MODE_CALC = 10       # Sets number of modes to run calc over 

# Time contraints and initial conditions ###
T1 = 5e-9                       # Time to run calc for (s)
DT = 1e-16                      # Time step for calc (s)
S0 = 0                          # Initial photon conc (cm^-3)
N0 = 1e16                       # Initial carrier conc (cm^-3)

# Multiprocessing settings ###
NUM_PROC = 6
p = mp.Pool(NUM_PROC)

###############################################################################
# Pre-calculations
###############################################################################
# Convertions to specific value ranges for this code ###
cur = CUR/1e3
cur_sweep_A = [i/1e3 for i in cur_sweep_mA]

# Single-mode operation values ###
if MM == 0:
    g_m = np.array([al_t])
    m0 = 0                          # Manually set m0 for single mode simulation
    wl_m = [LAM0]
    print('lam0=', LAM0, 'm')
    print('Gain m0 [Total loss]', al_t, 'cm^-1')


# Read in modal gain ###
if MM == 1:
    inp = np.load('INV_TMM.npz')
    wl_m = inp['WL']
    g_m = inp['gain']                 # Import gain in cm^-1
    m0 = int(np.ceil(len(g_m)/2))
    wl_m0 = wl_m[m0]
    d_wl = abs(np.average([wl_m[i]-wl_m[i-1] for i in range(1, len(wl_m))]))
    g_band = d_wl*M                     # Gain band FWHM (nm)
    # Set limit on modes in file to run calc over
    g_m = g_m[m0-MODE_CALC:m0+MODE_CALC]
    wl_m = wl_m[m0-MODE_CALC:m0+MODE_CALC]
    m0 = int(np.ceil(len(g_m)/2))       # Reset m0

# Simulation inputs/outputs ###
T = []                                # Time vector
y0 = [N0]                           # Initial conds [N]
for i in range(len(g_m)+1):         # Add initial S for each mode
    y0.append(S0)
print("m0 = ", m0)

###############################################################################
# Main function definition
###############################################################################
def solver(y, cur):                     # Input current & init conds

    out = []                            # Holder for all simulation outputs
    p = [cur, q, vol_cm, vg_cm, GAM]    # Parameters for odes

    # Setup integrator with desired parameters ###
    r = ode(laser_rates).set_integrator('dopri5', nsteps = 1e6)
    r.set_f_params(p).set_initial_value(y, 0)

    # Simulation run & check ###
    while r.successful() and r.t+DT < T1:
        r.integrate(r.t + DT)
        out.append(r.y)                   # Makes a list of 1d arrays
        T.append(r.t)

    out = np.array(out)                     # Convert from list to 2d array

    return out



# Define equations to be solved ###
def laser_rates(t, y, p):

    # Generate outputs for each mode ###
    dy = np.zeros([len(g_m)+2])

    # Carrier equation ###
    dy[0] = p[0]/(q*vol_cm) - coeff_gain(y[0], y[1], 0)*y[1]*vg_cm - tn(y[0])
    # Total carrier conc calc ###
    y[1] = sum([y[i] for i in range(2, len(dy))])    # Total stim emission
    # Calculation for each independent mode ###
    for i in range(len(g_m)):
        dy[i+2] = (GAM*coeff_gain(y[0], y[i], i)*vg_cm - 1/tp_m(g_m[i]))*y[i+2]\
        + GAM*R_SP
        #dy[i+2] = (GAM*coeff_gain(y[0], y[i], i)*vg_cm - 1/tp_m(g_m[i]))*y[i+2]\
        #+ GAM*spont(y[0], wl_m[i])

    # Display outputs of each mode ###
    #print(y)

    return dy


###############################################################################
# Supplementary definitions
###############################################################################
# Group index ###
def n_g(wl):
    x = IND_RID-(LAM0-wl)*D_IND
    #print('Group refractive index', x)
    return x


# Carrier decay rate (tn) removed above threshold ###
def tn(n):
    x =  A*n + B*(n**2) + C*(n**3)
    y = x*n 
    y  = "{:e}".format(y)
    #print('Carrier Recomb', y, 'ps^-1')
    return x


# Gain log calc p277 C&C, compensated for cavity ###
def coeff_gain(n, s, itr):  # s is modal total for photon conc
    x = (r_l*r_r)*(DG*n)*(1/(1+abs(itr-m0)/M**2))*(1/(1+EPS*s))
    y  = "{:e}".format(x)
    #print('Gain Factor', x,'cm^-1')
    return x


def spont_band(wl):
    x = 1/(1+(2*abs(LAM0-wl)/g_band)**2)
    y = "{:e}".format(x)
    #print('Spontaneous Band Factor', y)
    return x

# Coldren & Corzine appendix calculation laser-rate wiki spont factor ###
def beta_sp(wl):
    x = (2/np.pi)*GAM*(wl**4)/(8*np.pi*vol_nm*(IND_RID**2)*n_g(wl)*g_band)#\*spont_band(wl)
    #x = 1e-3
    #y = "{:e}".format(x)
    #print('Spontaneous emission factor', y)
    return x

# Approximation from Cassidy [1983] cavity amplification ###
def spont(n, wl):
    x = n*beta_sp(wl)*(1+(AL_CAV*cl_cm)/2)
    #y = "{:e}".format(x)
    #print('Spontaneous emission rate', y)
    return x

###############################################################################
# Plotting function definitions
###############################################################################
# Takes in Y array & returns desired data ###
def proc_li(y):
    y = np.array(y)
    col = [y[i][-1] for i in range(len(y))]
    car_conc = [col[i][0] for i in range(len(col))]
    pow_t = [p_mW(col[i][1]) for i in range(len(col))]
    pow_m = [p_mW(col[i][2:]) for i in range(len(col))]

    return pow_tot, pow_m


# Takes in Y array & returns desired data for dynamic calc ###
def proc_dynam_mm(y):
    y = np.array(y)
    car_conc = y[:, 0]
    pow_tot = p_mW(y[:, 1])
    pow_m = p_mW(y[:, 2:])
    pow_0 = p_mW(y[:, m0+1])
    y_end = y[-1, 1:]

    return car_conc, pow_tot, pow_m, pow_0, y_end


# Dynamic plotting ###
def plot_dynam_s(car_conc, phot_conc):

    f, axarr = plt.subplots(2, sharex=True)     # Two subplots
    axarr[0].plot(T, car_conc, 'g')
    axarr[0].set_ylabel("Carrier Conc ($cm^{-3}$)")
    axarr[0].set_title('Laser-Rate Simulation')
    axarr[1].plot(T, phot_conc, 'b')
    axarr[1].set_ylabel("Modal Power (mW)")
    axarr[1].set_xlabel("Time (s)")
    plt.show()

    return


# Dynamic plotting ###
def plot_dynam_mm(pow_m):
    plt.plot(T, pow_m)
    plt.ylabel("Power (mW)")
    plt.title('Laser-Rate Simulation')
    #plt.ylim(0, 3e16)
    plt.xlabel("Time (s)")
    plt.show()

    return


# Dynamic plotting ###
def smsr(pow_m):

    print(pow_m)
    # Steady-state SMSR ###
    plt.plot(wl_m, p_dBm(pow_m[-1]), '-*')
    plt.xlabel('WL (nm)')
    plt.ylabel('Power (dBm)')
    plt.title('Spectrum')
    plt.show()

    # Dynamic SMSR ###
    max_val = max(pow_m[-1])
    max_index = np.where(pow_m == max_val)
    sort = np.sort(pow_m[-1])
    val = sort[-2]
    ind = np.where(pow_m[-1]==val)
    pow_0 = [pow_m[i][m0] for i in range(len(T))]
    pow_nn = [pow_m[i][m0+1] for i in range(len(T))]
    pow_band = [pow_m[i][ind] for i in range(len(T))]
    smsr_nn = [p_dBm(pow_nn[i]) - p_dBm(pow_0[i]) for i in range(len(T))]
    smsr = [p_dBm(pow_band[i]) - p_dBm(pow_0[i]) for i in range(len(T))]
    plt.plot(T, smsr_nn, label='Nearest Neighbour')
    plt.plot(T, smsr, label='Across Band')
    plt.ylabel("Supression (dB)")
    plt.title('Dynamic SMSR')
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()

    return


def plot_li_s(pow_tot, pow_m):

    # Post solver calculations
    qe = [i/j for i, j in zip(pow_tot, cur_sweep_mA)]                # Convert to QE

    # Plotting two parameters on one plot ###
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(cur_sweep_mA, pow_tot, 'g-')
    ax2.plot(cur_sweep_mA, qe, 'b-')
    ax1.set_xlabel('Current (mA)')
    ax1.set_ylabel('Power (mW)', color='g')
    ax2.set_ylabel('Quantum Efficiency', color='b')
    plt.title("Single Mode LI")
    plt.show()

    plt.plot(cur_sweep_mA, pow_m)
    plt.title('Multimode LI')
    plt.xlabel('Current (mA)')
    plt.ylabel('Power (mW)')
    plt.show()

    return


###############################################################################
# Main section to call functions
###############################################################################
if __name__ == '__main__':
    tic = time.perf_counter()
    # Dynamic single mode ###
    if MM == 0 and LI == 0:
        res = solver(y0, CUR)
        n, pow_tot, pow_m, pow_0, y_end = proc_dynam_mm(res)
        toc = time.perf_counter()
        print(f"The code executed in {toc - tic:0.4f} seconds")
        plot_dynam_s(n, pow_0)

    # Dynamic multi-mode ###
    if MM == 1 and LI == 0:
       res = solver(y0, cur)
       #res = p.map(solver, y0, CUR)
       toc = time.perf_counter()
       print(f"The code executed in {toc - tic:0.4f} seconds")
       n, pow_tot, pow_m, pow_0, y_end = proc_dynam_mm(res)
       plot_dynam_mm(pow_m)
       smsr(pow_m)
       
       # Relaunch calc again at ss to extract LW data
       y_end[0] = n[-1]+n[-1]/10               # 10% modulation applied
       y_end = y_end.tolist()
       y_end.insert(1, 0)                      # Inset initial carrier conc
       T.clear()                            # Empty time list to reuse
       res = solver(y_end, cur)
       n, pow_tot, pow_m, pow_0, y_end = proc_dynam_mm(res)
       np.savez('Rate_out.npz', T, n, pow_0)
    
    # Steady-state single-mode LI ###
    if LI == 1:
        res = p.map(fn.partial(solver, y0), cur_sweep_A)
        m0, m = proc_li(res)
        toc = time.perf_counter()
        print(f"The code executed in {toc - tic:0.4f} seconds")
        plot_li_s(m0, m)
