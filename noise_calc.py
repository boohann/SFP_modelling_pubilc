'''
###############################################################################
######################### RIN/PN calculations in Python #######################
############################ Niall Boohan 2020 ################################
######################### niall.boohan@tyndall.ie ##############################
###############################################################################

BASED ON:
Theory and equations sourced from:
Title: Diode Lasers and Photonic Integrated Circuits
Author: Larry Coldren, Milan Mashanovitch, and Scott W. Corzine
Year: 2012

NOTES:
- STILL UNDER DEVELOPEMENT!!!!!

FUTURE REVISIONS:
- Calculate ni and n0 from rate-equation model
- Fundamental calculation of gain compression
'''

# Import necessary libraries ###
import numpy as np
import matplotlib.pyplot as plt

# Load imputs ###
EX = np.load('Extract.npz')

# Import parameters ###
from fileinput import CUR, GAM,  DG, ap, IND_RID, h, c, q, vg_cm, vol_cm,\
    t_dn, R_SP

tn = 2

# Input values from Rate-eqations model ###
S = 1.98e16                         # Photon conc (cm^-3)
P0 = 6.5                            # Power output (mW)
I_th = 16                           # Threshold current (mA)

# Conversion ###
I = CUR/1e3
P0 = P0/1e3
I_th = I_th/1e3

### Simualtion input  parameters ###
damp = EX['tau']            # Damping rate (ns)
osc_freq = EX['freq']       # Relaxation oscillation freq (GHz)
tn  = tn*1.0e-9             # Carrier relaxation time in seconds (s)
gam = 1/(damp*1e-9)         # Damping rate (s^-1)
n0 = 0.45                   # slope efficiency
ni = 1                      # Injection efficiency



# Define freq span ###
freq = np.linspace(0, 1e11, 1000000)

# Resonant freq ###
om_r_sq = (osc_freq*2*np.pi)**2

# Schawlow-Townes linewidth ###
nu_st = (GAM*R_SP)/(4*np.pi*S)
print('Schalow-Townes linewidth =', nu_st/1e6, 'MHz')

# Plot Lorentzian linewidth ###
lor_lw = 1/(2*np.pi*damp)
print('Lorentz linewidth=', lor_lw, 'GHz')
lor_freq = np.linspace(int(osc_freq-lor_lw*2), int(osc_freq+lor_lw*2), 1000)
line_shape = [2*np.pi*(lor_lw/(4*(lor_freq[i]/1e9-osc_freq)**2+lor_lw**2))
              for i in range(len(lor_freq))]

#plt.plot(lor_freq, line_shape)
#plt.show()

# Calc stimulated current ###
I_st = ni*(I-I_th)

# Angular frequency ###
def omega(f):
    return 2*np.pi*f


# Modulation transfer function ###
def H(f):
    om = omega(f)
    return (om_r_sq)/(om_r_sq-om**2+(om*gam)*1j)


# a Factor 1 ###
def a1(f):
    return (8*np.pi*nu_st*P0)/(h*f)*(1/(t_dn**2))+\
            n0*(om_r_sq**4)*(ni*(I-I_th)/I_st-1)

# a Factor 2 ###
def a2(f):
    return (8*np.pi*nu_st*P0)/(h*f)-2*n0*om_r_sq*(GAM*ap/DG)

# Returns RIN/(Delta f) ###
def RIN(f):
    return (2*h*f/P0)*(((a1(f)+a2(f)*(omega(f))**2)/om_r_sq**2)*H(f)**2+1)

def PN(f):
    return (h*f*P0)*(((a1(f)+a2(f)*(omega(f))**2)/om_r_sq**2)*H(f)**2+1)

if __name__ == '__main__':

    IN = [10*np.log10(RIN(freq[i])) for i in range(len(freq))]
    IM = [10*np.log10(PN(freq[i])) for i in range(len(freq))]

    f, axarr = plt.subplots(2, sharex=True)     # Two subplots
    axarr[0].semilogx(freq/1e9, IN, 'g-')
    axarr[0].set_ylabel('Intensity (dBm/Hz)', color='g')
    axarr[0].set_title('Noise')
    axarr[0].set_xscale('log')
    axarr[1].semilogx(freq/1e9, IM, 'b-')
    axarr[1].set_ylabel('Phase (dBm/Hz)', color='b')
    axarr[1].set_xlabel("Freq (Hz)")
    plt.show()
#om_r_sq = ((Gam*v_g*a)/(q*V))*ni*(I-I_th)
