# -*- coding: utf-8 -*-

'''
This file reads the simulation data generated by 'plate.py' and 
displays the relevant quantities (transverse displacement, spectrum, 
constraint forces).
'''

# LIBRARIES -------------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib.animation as mpla
import numpy as np
import scipy.signal as sig
import os

os.chdir(os.getcwd()+'/..')

import src.UK as UK
import src.data as data
import src.disp as disp

# MAIN SCRIPT -----------------------------------------------------------------

# Simulation folder's number
n = 6

# Get the simulation data
M, K, C, A, b, W, V, Wc, Vc, phi, phi_c, Fext, Fext_phys, x, xd ,xdd, Fc,\
    Fc_phys, h = data.load_simulation(n)

t = np.arange(0, h*x.shape[0], h)

idx_Fext = 55

X = np.fft.rfft(x[:,idx_Fext])
f = np.fft.rfftfreq(x[:,idx_Fext].size, h)

# Plot the relevant data
disp.set_gui_qt()

plt.figure()
plt.title('Board\'s displacement at the excitation')
plt.plot(t, x[:,idx_Fext])
plt.xlabel('$t$ [s]')
plt.ylabel(r'$y_s$ [m]')
plt.show()

plt.figure()
plt.title('Board\'s displacement at the corner')
plt.plot(t, x[:,-1])
plt.xlabel('$t$ [s]')
plt.ylabel(r'$y_s$ [m]')
plt.show()

plt.figure()
plt.title('Excitation')
plt.plot(t,Fext_phys[:,idx_Fext].toarray())
plt.show()

plt.figure()
plt.title('Board\'s displacement\'s spectrum magnitude at the excitation')
plt.plot(f, 20*np.log(np.abs(X)))
plt.xlabel(r'$f$ [Hz]')
plt.ylabel(r'$|X_s|$')
plt.show()