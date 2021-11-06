###############################################################################
########################### Script of input values ############################
############################## Calcs & settings ###############################
###############################################################################

import numpy as np

# Settings
COAT = 0
# Turn on and off facet coating
AUTO_SLOT = 1       # Automatic slot width calc
slot_order = 0      # Slot order size (steps of 1/2)
ALIGN = 0           # Cos align = 0, sin align = 1

# Dimensions ###
CL = 275            # Cavity length (um)
cl_m = CL*1e-6      # Cavity length um->m
cl_cm = CL*1e-4     # Cavity length um->cm
W = 3               # Cavity width (um)
T = 7               # Cavity thickness (nm)
GAM = 0.015*3        # Modal confinement factor, same for all modes, Coldren


# Currents ###
CUR = 20                                 # Pumping current (mA)
cur_sweep_mA = np.linspace(0, 100, 5)    # Current sweep (mA)


# Refractive indexes ###
IND_RID = 3.2#3.217#3.1811#     # Effective refractive index cavity
IND_DEL = 0.01#0.027#0.0197#    # Refractive index step
D_IND = 0.005                   # Change in refractive index per (nm)
AL_CAV = 0                      # Cavity loss (cm^-1)
D_AL = 0.025                    # Change in loss with WL (dB/nm)
NO_SLOT = 20                    # No of slots
ind_def = IND_RID-IND_DEL       # Refractive index slot

# Chosen WL ###
LAM0 = 1550.0                   # Chosen centre WL 1st run for calc & gain peak (nm)
lam0_m = LAM0*1e-9              # Convert lam0 to m
lam0_um = LAM0*1e-3             # Convert lam0 to um


# Auto create slotwidth as a fract of WL ###
if AUTO_SLOT == 1:
    sw = ((slot_order/2)*4 + 1)*(((LAM0)/1e9)/(4*ind_def))
    #sw = 9*(((LAM0)/1e9)/(4*ind_def))
    #fid = ((slot_order/2)*4 + 1)*((-200/1e9)/(4*ind_def))  
    #fid = ((slot_order/2)*4 + 1)*((+70/1e9)/(4*ind_def)) 
    #fid = ((slot_order/2)*4 + 1)*((-110/1e9)/(4*ind_def)) 
    fid = ((slot_order/2)*4 + 1)*((-13/1e9)/(4*ind_def)) 
    #fid = 0
    sw = sw+fid
else:
    fid = 0


if AUTO_SLOT == 0:
    sw = 1.1/1e6      # Set arbitrary slot size

m0 = 2*(cl_m*IND_RID-NO_SLOT*sw*IND_DEL)/lam0_m
m0 = round(m0) 
sw_cm = sw*1e2              # Convert sw (cm)
sw_um = sw*1e6
fid_um = fid*1e6

# Reflectivity, !!sqrt if power reflectance from TMM!! ###
if COAT == 0:
    r_l = r_r = (IND_RID-1)/(IND_RID+1)     # Calc ref from cacity
if COAT == 1:
    r_l = np.sqrt(0.936)#np.sqrt(0.70223)#np.sqrt(0.884114)     # Left cavity ref
    r_r = 0.52572919136         # Right cavity ref

# Material properties ###
A = 0                   # Carrier loss assumed negligible
B = 0.8e-10             # Bimolecular recomb (cm^3s^-1)
C = 3.5e-30             # Auger recom (cm^6s^-1)
M = 25                  # Gain band FWHM (no of modes)

# Cavity properties ###
DG = 5.34e-16           # Differential gain (cm^2)
EPS = 1.5e-17           # Gain compression factor 3D (cm^3)
ap = 2.37e-14           # Gain compression 2D (cm^2)
t_dn = 1.57e-9          # Carrier diff lifetime (s)
R_SP = 0.3e21           # Spont emission (cm^-3/s)

# Convert to cm ###
cl_cm = CL/1e4
w_cm = W/1e4
t_cm = T/1e7


# Physical contants ###
q = 1.60217663e-19      # Electron charge (C)
h = 6.62607004e-34      # Plank's contant (Js)
c = 2.99792458e8        # SOL (ms^-1)

# Calc device volume (cm^3) ###
vol_cm = cl_cm*t_cm*w_cm
vol_nm = vol_cm*1e21

print(f"Cavity volume =  {vol_cm:0.2e} cm^3")
print(f"Modal volume =  {vol_cm*GAM:0.2e} cm^3")

# Centre frequency ###
f = c/LAM0          # Frequency m0 (GHz)
f_Hz = f*1e9
print(f"Selected wavelength =  {LAM0:0.4f} nm")
print(f"Selected frequency =  {f/1e3:0.4f} THz")

# Loss & photon lifetime ###
al_mir = (1/cl_cm)*np.log(1/(r_l*r_r))
al_t = AL_CAV + al_mir
print(f"Cavity loss =  {AL_CAV:0.4f} cm^-1")
print(f"Mirror loss =  {al_mir:0.4f} cm^-1")
print(f"Total loss =  {al_t:0.4f} cm^-1")

# Cavity effective index ###
ind_eff = IND_RID-IND_DEL*(NO_SLOT*sw_um)/CL

# Group velocity & lifetime ###
vg = c/ind_eff               # (ms^-1)
vg_cm = vg*100               # (cms^-1)
print(f"Group velocity =  {vg_cm:0.4e} cm/s")


###############################################################################
# Functions
###############################################################################
# Convert loss/gain per unit len to imaginary refractive index ###
def kappa(wl, al):
    return ((wl*1e2)*al)/(4*np.pi)


# Group index ###
def n_group(wl, dn, n):
    return wl*dn+n


# Power conversion mW ###
def p_mW(s):
    return h*f_Hz*((s*vol_cm)/tp_m(-al_t))*1e3


# Power conversion dBm ###
def p_dBm(mW):
    return 10*np.log10(mW)


# Fresnel reflection calc ###
def fres(n1, n2):
    return (n1-n2)/(n1+n2)


# Parabolic falloff gain factor ###
def g_prof_fact(wl):
    return 1/(1+abs(wl-LAM0)/M**2)


# Apply linearly increasing loss to cavity ###
def lin(wl, al):
    return al-(LAM0-wl)*D_AL


# tp_m  varying loss for each mode, comp for cavity  ###
def tp_m(g):
    al_c = AL_CAV
    al_g = np.exp(g*cl_cm)
    al_t = (1 - al_g)/cl_cm + AL_CAV
    x = 1/(vg_cm*al_t)
    #y  = "{:e}".format(x)
    #print('Photon lifetime = ', y,'s')
    return x
