'''
###############################################################################
############################ Program to SMM Python ############################
############################# Niall Boohan 2020 ###############################
########################## niall.boohan@tyndall.ie ############################
###############################################################################

BASED ON:
    - Raymond C. Rumpf at the course Computational EM:
        https://empossible.net/academics/emp5337/
- Sathyanarayan Rao from https://www.mathworks.com/matlabcentral/fileexchange/
47637-transmittance-and-reflectance-spectra-of-multilayered-dielectric-stack-
using-transfer-matrix-method
- TMM calc approximation from www.batop.de

NOTES:
- TE plane wave assumed no kx or ky components
- Updated to Python and applied to repeating laser Bragg grating
'''

###############################################################################
# Dashboard
###############################################################################
# Import libraries ###
import scipy.linalg as la
import numpy as np
import multiprocessing as mp
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import time


# Import parameters ###
from fileinput import CL, sw_um, IND_RID, ind_def, lam0_um, ALIGN

# Number of process and time
tic = time.perf_counter()
num_processes = 7

# Device parameters load from inv code ###
IN = np.load('InvGen.npz')
four_min = -min(IN['gain'])  # +2
mode = IN['gain']
mode_wl = IN['WL']
val_gain = four_min     # Set TMM gain to arbitary value for GT = 0


# Scan resolution values
low_res = 125                   # points per nm
high_res = low_res*10

# Errors grating position & cavity length ###
ERROR_GRAT = 0              # Turn on error in grating position
ERROR_LEN = 0               # Turn on error in cavity length 1 = RHS, 2 = LHS
err = 1*lam0_um/(32*IND_RID)     # 1/24 lambda0 error

if ERROR_LEN == 1 or ERROR_LEN == 3:
    CL = CL + err

# Settings ###
WL_start = 1510                 # Start WL (nm)
WL_stop = 1590                  # Stop WL (nm)
WL_step = 3*15000              # Number of steps for WL scan

# Code Settings ###
fac_only = 0                    # Load cavity layers0
AF = 0                          # 0 No facet, 1 LHS, 2 RHS, 3 LHS&RHS
GT = 1                          # Turn on threshold gain calc

# Gain to complex ref ###
LO = 1                                     # 0 no loss, 1 WL indep, !2 WL dep!

g_range = np.arange(-31.15, -31.05, 0.01)
d_def = 0                                   # Add loss defect (cm^-1)

# Generic layers ###
n_vac = 1+0j
n_inc = n_vac
n_med = IND_RID+0j
e_med = n_med**2
if fac_only == 0:
    e_l = n_inc**2
    e_r = n_inc**2  # (2.14+0j)**2
if fac_only == 1 and AF == 1:
    e_r = n_med**2
    e_l = n_inc**2
if fac_only == 1 and AF == 2:
    n_inc = n_med
    e_l = n_med**2
    e_r = n_vac**2
u = (1+0j)**2           # Mag constant throughout structure
print('n incident =', n_inc)
print('n medium =', n_med)
print('n vaccum = ', n_vac)


###############################################################################
# Cavity calculations
###############################################################################
# Applying calculation matrices ###
wl = np.linspace(WL_start, WL_stop, WL_step)   # Free space wavelength
wl_m = [wl[i]*1e-9 for i in range(len(wl))]

# Homogenous calcs ###
kz = np.sqrt((n_vac**2)*u)          # Apply refractive ind of incident material
Id = np.identity(2)                 # Creating identity matrix
Q = np.array([[0, 1], [-1, 0]])     # Vac medium same as external L&H
Omh = 1j*kz*Id                      # Omega
Vh = np.dot(Q, np.linalg.inv(Omh))

if fac_only == 0:
    # Load slot positions from external calculation file ###
    slot_pos = IN['pos_slot']
    print(slot_pos)
    if ERROR_GRAT == 1:
        slot_pos = [slot_pos[i] - err for i in range(len(slot_pos))]
    lr = [slot_pos[i]-slot_pos[i-1]-sw_um for i in range(1, len(slot_pos))]
    # Manual edit slot positions ###
    # lr = [slot_pos[i]-slot_pos[i-1]-sw_um-fid_um*(3.2/4)
    #          for i in range(1, len(slot_pos))]
    if ALIGN == 0:
        lr.insert(0, slot_pos[0])
    if ALIGN == 1:
        lr.insert(0, slot_pos[0]-sw_um/2)
    ls = [sw_um for i in range(len(slot_pos))]
    len_list = [None]*(len(lr)+len(ls))
    len_list[1::2] = ls
    len_list[::2] = lr

    # Appends last section to laser ###
    if ALIGN == 0:
        len_list.append(CL-slot_pos[-1]-sw_um)
    if ALIGN == 1:
        len_list.append(CL-slot_pos[-1]-sw_um/2)

    # Add error position to LHS if activated ###
    if ERROR_LEN == 2 or ERROR_LEN == 3:
        len_list[0] = len_list[0] + err

    # Convert lengths to meters before attaching facets ###
    print(len_list)
    print('Confirm length = ', sum(len_list))
    l_m = np.empty(len(len_list))
    for i in range(len(len_list)):
        l_m[i] = len_list[i]*1e-6
    # l_m = np.array([500*1e-6])


if fac_only == 1:
    l_m = []; ER = []; UR = []


# Convert loss/gain per unit len to imaginary refractive index ###
def kappa(WL, alpha):
    return ((WL*1e2)*alpha)/(4*np.pi)


def n_real(WL, dn, n):
    return WL*dn+n


# Creats dielectric stack for inputs
def di_sta(gain_applied, disp, wl):
    if AF == 1 or AF == 3:
        fac_l = np.load('FAC_L.npz')
        n_l = fac_l['ind_list']
        len_l = fac_l['length_list']
        l1 = len(len_l)
        l2 = 0
    if AF == 2 or AF == 3:
        fac_r = np.load('FAC_R.npz')
        n_r = fac_r['ind_list']
        len_r = fac_r['length_list']
        l2 = len(len_r)
        l1 = 0
        len_l = 0
    if AF == 0:
        l1 = 0
        l2 = 0
    if fac_only == 1:
        l0 = 0
    if fac_only == 0:
        l0 = len(l_m)
    arr = np.empty((len(wl), 4, l0+l1+l2), dtype=object)
    arr[:, 2, :] = u

    for i in range(len(wl)):
        if AF == 1 or AF == 3:
            E0 = [(n_real(wl[i], 0, n_l[j])+1j*kappa(wl[i], 0))**2
                  for j in range(len(n_l))]
        if fac_only == 0:
            E1 = (n_real(wl[i], 0, IND_RID)+1j*kappa(wl[i], gain_applied))**2
            E2 = (n_real(wl[i], 0, ind_def)+1j*kappa(wl[i], gain_applied))**2
        if AF == 2 or AF == 3:
            E3 = [(n_real(wl[i], 0, n_r[j])+1j*kappa(wl[i], 0))**2
                  for j in range(len(n_r))]
        if fac_only == 0:
            arr[i, 1, ::2] = E1
            arr[i, 1, 1::2] = E2
        if AF == 1 or AF == 3:
            arr[i, 1, :len(E0)] = E0
        if AF == 2 or AF == 3:
            arr[i, 1, -len(E3):] = E3
    if AF == 1 or AF == 3:
        arr[:, 3, 0:l1] = len_l
    arr[:, 3, l1:len(l_m)+l1] = l_m
    if AF == 2 or AF == 3:
        arr[:, 3, l1+len(l_m):] = len_r
    for i in range(len(wl)):
        arr[i, 0] = wl[i]
    return arr


# Pass SMM a range of wavelength and an array of matrices for each sect ###
def calc(M):
    # M[0] = WL, M[1] = E, M[2] = U, M[3] = L
    # Initialaze global scattering matrix ###
    Sg11 = np.zeros(shape=(2, 2))
    Sg12 = Id
    Sg21 = Id
    Sg22 = np.zeros(shape=(2, 2))

    # Wave vector wavespace ###
    k0 = (2*np.pi)/M[0][0]

    # Add left external media ###
    krz = np.sqrt(e_l*u)
    Omr = 1j*krz*Id
    Vr = np.dot(Q, np.linalg.inv(Omr))
    Ar = Id + np.dot(np.linalg.inv(Vh), Vr)
    Br = Id - np.dot(np.linalg.inv(Vh), Vr)
    Sr11 = -1*Id*np.dot(np.linalg.inv(Ar), Br)
    Sr12 = 2*Id*np.linalg.inv(Ar)
    Sr21 = 0.5*Id*(Ar - np.linalg.multi_dot([Br, np.linalg.inv(Ar),  Br]))
    Sr22 = np.dot(Br, np.linalg.inv(Ar))

    # Connect external reflection region to device ###
    Sg11, Sg12, Sg21, Sg22 = \
        redhefferstar(Sr11, Sr12, Sr21, Sr22, Sg11, Sg12, Sg21, Sg22)

    # Reflection + phase change for each layer for every wavelength ###
    for j in range(len(M[3])):
        kz = np.sqrt(M[1][j]*M[2][j])
        Om = 1j*kz*Id
        V = np.dot(Q, np.linalg.inv(Om))
        A = Id + np.dot(np.linalg.inv(V), Vh)
        B = Id - np.dot(np.linalg.inv(V), Vh)
        X = la.expm(Om*k0*M[3][j])
        D = A - np.linalg.multi_dot([X, B, np.linalg.inv(A), X, B])
        S11 = np.dot(np.linalg.inv(D),
                     np.linalg.multi_dot([X, B, np.linalg.inv(A), X, A])-B)
        S22 = S11
        S12 = np.linalg.multi_dot([
            np.linalg.inv(D), X,
            A-np.linalg.multi_dot([B, np.linalg.inv(A), B])])
        S21 = S12
        # Update gloabal S-matrix
        Sg11, Sg12, Sg21, Sg22 =\
            redhefferstar(Sg11, Sg12, Sg21, Sg22, S11, S12, S21, S22)

    # Add right external media ###
    ktz = np.sqrt(e_r*u)
    Omt = 1j*ktz*Id
    Vt = np.dot(Q, np.linalg.inv(Omt))

    At = Id + np.dot(np.linalg.inv(Vh), Vt)
    Bt = Id - np.dot(np.linalg.inv(Vh), Vt)
    St11 = np.dot(Bt, np.linalg.inv(At))
    St12 = 0.5*Id*(At - np.linalg.multi_dot([Bt, np.linalg.inv(At), Bt]))
    St21 = 2*Id*np.linalg.inv(At)
    St22 = -1*Id*np.dot(np.linalg.inv(At), Bt)
    Sf11, Sf12, Sf21, Sf22 =\
        redhefferstar(Sg11, Sg12, Sg21, Sg22, St11, St12, St21, St22)

    # Source ###
    k_inc = np.array([0, 0, k0*n_inc])
    aTE = np.array([0, 1,  0])
    aTM = np.cross(aTE, k_inc)/np.linalg.norm(np.cross(aTE, k_inc))
    pte = 1/np.sqrt(2)
    ptm = pte*1j
    P = pte*aTE + ptm*aTM
    c_inc = np.array([[P[0]], [P[1]]])

    # Tranmitted and reflected fields ###
    Er = np.dot(Sf11, c_inc)
    Et = np.dot(Sf21, c_inc)
    Erx = Er[0]
    Ery = Er[1]
    Etx = Et[0]
    Ety = Et[1]

    # Transmittance and reflectance ###
    R = abs(Erx)**2 + abs(Ery)**2
    T = (abs(Etx)**2 + abs(Ety)**2)*np.real((u*ktz)/(u*krz))
    return R, T


def const(G, WL):
    p = mp.Pool(num_processes)
    M = np.empty(len(WL))

    # No loss, no dispersion ###
    if LO == 0:
        M = di_sta(0, 0, WL)
        TMM = p.map(calc, M)

    # Constant Loss/Gain
    if LO == 1:
        M = di_sta(G, 1, WL)
        TMM = p.map(calc, M)
        p.close()
        p.join()

    # Loss & dispersion ###
    if LO == 2:
        M = di_sta(G, 0, WL)
        TMM = p.map(calc, M)
        p.close()
        p.join()
    return TMM


# Function to find threshold gain of each mode ###
def threshold_scan(scan_value, wl_range):
    pow_lst = []
    tmm = const(scan_value, wl_range)
    T = np.empty(len(wl_range))
    for i in range(len(wl_range)):
        T[i] = tmm[i][0]
    max_value = max(T)
    max_index = np.where(T == max_value)
    peak = wl_range[max_index[0][0]]
    i = four_min+2
    pow_out = 0
    while pow_out < 10000 and i > four_min*2:
        tmm = const(i, [peak])
        pow_out = tmm[0][0][0]
        i = i-0.01
        pow_lst.append(pow_out)
    print(pow_lst)
    max_val = max(pow_lst)
    threshold = i
    return threshold, peak, max_val


###############################################################################
# Combining matrices function, Redheffer
###############################################################################
def redhefferstar(SA11, SA12, SA21, SA22, SB11, SB12, SB21, SB22):
    Num = len(SA11)
    Id = np.identity(Num)
    DA = np.dot(SA12, np.linalg.inv(Id-np.dot(SB11, SA22)))
    FA = np.dot(SB21, np.linalg.inv(Id-np.dot(SA22, SB11)))
    SAB11 = SA11 + np.linalg.multi_dot([DA, SB11, SA21])
    SAB12 = np.dot(DA, SB12)
    SAB21 = np.dot(FA, SA21)
    SAB22 = SB22 + np.linalg.multi_dot([FA, SA22, SB12])
    return SAB11, SAB12, SAB21, SAB22


######################################################################
# Main section to call functions
######################################################################
if __name__ == '__main__':
    if GT == 0:
        tmm = const(val_gain, wl_m)
        R = np.empty(len(wl_m))
        T = np.empty(len(wl_m))
        for i in range(len(wl_m)):
            R[i] = tmm[i][0]
            T[i] = tmm[i][1]
        plt.plot(wl_m, R)
        plt.title('Reflectance Power')
        plt.ylabel('Arbitary optical power')
        plt.xlabel('WL (nm)')
        plt.show()
        plt.plot(wl_m, T)
        plt.title('Transmission Power')
        plt.ylabel('Arbitary optical power')
        plt.xlabel('WL (nm)')
        plt.show()

    # Gain threshold ###
    if GT == 1:
        T = np.empty(len(wl_m))
        threshold = []
        wavelength = []
        tmm = const(four_min, wl_m)
        for j in range(len(wl_m)):
            T[j-1] = tmm[j-1][1]
        # plt.plot(R)
        # plt.show()
        peaks, _ = find_peaks(T)
        wl_peak = np.array(wl_m)[peaks]
        sep_peak = np.average([wl_peak[i]-wl_peak[i-1]
                               for i in range(1, len(wl_peak))])
        wl_pack = [np.arange(i-0.3*sep_peak, i+0.3*sep_peak, 1e-13)
                   for i in wl_peak]
        for i in range(len(wl_pack)):
            j = four_min+2
            pow_out = 0
            pow_prev = 0
            avg_lst = [1, 1, 1]
            k = 1
            while j > four_min*2:
                thresh, wl_point, pow_max = threshold_scan(j, wl_pack[i])
                print('j here =', j)
                print('Wavelength point =', wl_point)
                print('threshold =', thresh)
                if thresh < four_min*0.9 and thresh > four_min*1.95:
                    threshold.append(thresh)
                    wavelength.append(wl_point)
                    break
                if pow_max < pow_prev:
                    k = -k/1.5
                else:
                    j -= 1*k
                pow_prev = pow_max
                avg_lst.append(pow_max)
                avg_lst.pop(0)
                if pow_max > 0.95*np.average(avg_lst) and pow_max\
                   < 1.05*np.average(avg_lst):
                    k = k*1.3
        toc = time.perf_counter()
        print(f"The code executed in {toc - tic:0.4f} seconds")
        threshold = [i*-1 for i in threshold]
        wavelength = [i*1e9 for i in wavelength]
        np.savez('INV_TMM.npz', WL=wavelength, gain=threshold)
        print(threshold)
        print(wavelength)
        plt.plot(wavelength, threshold, '-*')
        plt.title('Threshold TMM')
        plt.xlabel('$\lambda$ (nm)')
        plt.ylabel('$\Delta$G$_{th}$ (cm$^{-1}$)')
        plt.show()
