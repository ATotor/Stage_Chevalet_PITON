# -*- coding: utf-8 -*-

'''
This file reads the simulation data generated by 'main_toy2el' and 
displays the relevant quantities (transverse displacement, spectrum, 
constraint forces).
'''

# LIBRARIES -------------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib.animation as mpla
import numpy as np
import os

os.chdir(os.getcwd()+'/..')

import src.UK as UK
import src.data as data
import src.disp as disp

# MAIN SCRIPT -----------------------------------------------------------------

# Simulation folder's number
n = 12

# Get the simulation data
M, K, C, A, b, W, V, Wc, Vc, phi, phi_c, Fext, Fext_phys, x, xd ,xdd, Fc,\
    Fc_phys, h = data.load_simulation(n)

t = np.arange(0, h*x.shape[0], h)

idx_Fext = 89

# Plot the relevant data
disp.set_gui_qt()

plt.figure()
plt.plot(t, x[:,idx_Fext])
plt.show()

plt.figure()
plt.plot(t,Fext_phys[:,idx_Fext].toarray())
plt.show()

plt.figure()
plt.plot(t,Fc_phys)
plt.show()

# %%

animate = True

if animate:
    Nt  = t.size
    Nxs = 100 
    L   = 0.65
    xs  = np.linspace(0, L, Nxs)
    
    plt.figure()
    line, = plt.plot(xs, x[0,:Nxs])
    plt.ylim(-5e-3, 5e-3)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('t = 0 s')
    
    def animate(n, xs, x):
        speed = 5
        plt.title(f'String : t = {t[speed*n]:.3f} s')
        line.set_ydata(x[speed*n,:Nxs])
        
    n       = np.arange(0,Nt)
    anim    = mpla.FuncAnimation(plt.gcf(), animate, n, fargs = (xs, x), 
                                 interval = 1)
    plt.show()