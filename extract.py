'''
###############################################################################
######################### Extract decay constant from #########################
############################### rate-eqation model ############################
############################## Niall Boohan 2020 ##############################
########################### niall.boohan@tyndall.ie ###########################
###############################################################################


NOTES:
    - Fit Fourier to N for carrier decay time!!!
    - Should this be E-field?

'''

# Inport libraries ###
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import lmfit as lm


# Import data ###
data = np.load('Rate_out.npz')
S = data['arr_2']
T = data['arr_0']
T = [T[i]/1e-9 for i in range(len(T))]
S = [S[i] for i in range(len(S))]

print(S)
print(T)
# Fitting ###
model = lm.models.ExpressionModel("ampl * cos((x - x0)*freq) * exp(-x/tau) + offset")
params = model.make_params(ampl=max(S), x0=0, freq=10, tau=1,
                       offset=S[-1])
fit = model.fit(S, params, x=T)
tau = fit.params["tau"].value
freq = fit.params["freq"].value
print(fit.fit_report())

# Save outputs ###
np.savez('Extract.npz', tau=tau, freq=freq)

# Plotting ###
f = plt.figure()
ax = f.add_subplot(111)
ax.plot(T, fit.best_fit, 'go')
ax.plot(T, S, 'b')
ax.text(0.6, 0.6, r"Oscillation freq = {0} GHz" "\n" r"$\tau$={1} ns"
        .format(round(freq, 3), round(tau, 3)),
        horizontalalignment='center',
        verticalalignment='center',
        transform = ax.transAxes
         )
ax.set_xlabel('Time (ns)')
ax.set_ylabel('Modal power (mW)')
plt.title('Decay & oscillation fitting')
plt.show()
