# -*- coding: utf-8 -*-

'''
This file reads the simulation data generated by 'main_2pol' and 
displays the relevant quantities (transverse displacement, spectrum, 
constraint forces).
'''

# LIBRARIES -------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import os

os.chdir(os.getcwd()+'/..')

import src.UK as UK
import src.data as data
import src.disp as disp

# MAIN SCRIPT -----------------------------------------------------------------

# Simulation folder's number
n = 4

# Get the simulation data
M, K, C, A, b, W, V, Wc, Vc, phi, phi_c, Fext, Fext_phys, x, xd ,xdd, Fc,\
    Fc_phys, h = data.load_simulation(n)

t = np.arange(0, h*x.shape[0], h)

# Spectra computations
Xf_v = np.fft.rfft(x[:,0])
Xf_h = np.fft.rfft(x[:,2])
Xb_v = np.fft.rfft(x[:,1])
Xb_h = np.fft.rfft(x[:,3])
f = np.fft.rfftfreq(x[:,0].size,h)
f_idx = f<1000


# Plot the relevant data
AXIS_SIZE   = 15
TITLE_SIZE  = 20

disp.set_gui_qt()

plt.figure(figsize=(15,10))
grid = plt.GridSpec(2, 1, wspace=0.4, hspace=0.6)

plt.subplot(grid[0, 0])
plt.plot(t, x[:,0], color="blue")
plt.plot(t, x[:,1], color="red")
plt.plot(t, x[:,2], color="blue",linestyle='--')
plt.plot(t, x[:,3], color="red",linestyle='--')
plt.xlabel("Time [s]", fontsize=AXIS_SIZE)
plt.ylabel(r"$x_S$(t) [m]", fontsize=AXIS_SIZE)
plt.title("String response - Waveform", fontsize=TITLE_SIZE)
#plt.ticklabel_format(axis='y', style='sci', scilimits=(-3,-3))
plt.ylim([-0.015, 0.01])

plt.subplot(grid[1, 0])
plt.loglog(f[f_idx], np.abs(Xf_v[f_idx]), color="blue", label="Finger, vert")
plt.loglog(f[f_idx], np.abs(Xb_v[f_idx]), color="red", label="Bridge, vert")
plt.loglog(f[f_idx], np.abs(Xf_h[f_idx]), color="blue", linestyle="--", 
           label="Finger, hor")
plt.loglog(f[f_idx], np.abs(Xb_h[f_idx]), color="red", linestyle="--", 
           label="Bridge, hor")
plt.xlabel("Frequency [Hz]", fontsize=AXIS_SIZE)
plt.ylabel(r"$X_S$(f)", fontsize=AXIS_SIZE)
plt.legend()
plt.title("String response - spectrum", fontsize=TITLE_SIZE)
plt.ylim([1e-3, 1e3])