###############################################################################
##################### Code for calculating Bragg mirrors ######################
############################# Niall Boohan 2020 ###############################
############################# niall.boohan@tyndal.ie ##########################
###############################################################################


'''
SOURCES: www.batop.de
'''

# Import libraries ###
import numpy as np
import matplotlib.pyplot as plt

# Import parameters ###
from fileinput import LAM0, lam0_m, IND_RID

# Import functions ###
from fileinput import fres

# Settings ###
PLOT = 0        # Turn on plotting of range
SIDE = 0        # 0 LHS, 1 RHS

# Inputs ###
X = 3.2
LAYERS = 1     # Define number of pairs
N = LAYERS*2+1  # Number of Layers
n_h = X+2
n_l = X-2#0.2#0.5#0.9
n_0 = 1+0j
app_r = 0.2
TICK = 1        # 1 HR, 2AR
in_lam0 = 1550.287352490166/1e9

def r(n):
    return (IND_RID - n)/(IND_RID + n)

# Permeability
u = (1+0j)**2
U = [u, u]

if TICK == 1:
    if SIDE == 0:
        ind = [n_h, n_l]
        length = [in_lam0/(4*n_h), in_lam0/(4*n_l)]
    if SIDE == 1:
        ind = [n_l, n_h]
        length = [in_lam0/(4*n_l), in_lam0/(4*n_h)]
    ind_list = np.tile(ind, LAYERS)
    perm_list = np.tile(U, LAYERS)
    length_list = np.tile(length, LAYERS)
    #length_list = np.insert(length_list,0,in_lam0/(4*n_l))
    #ind_list = np.insert(ind_list,0,n_l)
    #perm_list = np.append(perm_list, u)
    #perm_list = np.append(perm_list, 1)
if TICK == 2:
    n = np.sqrt(n_0*IND_RID)
    #n = 2.193
    ind_list = [n]
    perm_list = [1]
    length_list = [lam0_m/(4*n)]

print(perm_list)
print(ind_list)
print(length_list)

if SIDE == 0:
    np.savez('FAC_L.npz', ind_list=ind_list, perm_list=perm_list
             , length_list=length_list)

if SIDE == 1:
    np.savez('FAC_R.npz', ind_list=ind_list, perm_list=perm_list
             , length_list=length_list)


if PLOT == 1:
    n_lst = np.linspace(1, 5, 100)
    plt.plot(n_lst, r(n_lst))
    plt.xlabel('Refractive index')
    plt.ylabel('Reflectivity')
    plt.show()
