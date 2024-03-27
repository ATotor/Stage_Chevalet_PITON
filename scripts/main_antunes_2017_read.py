# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os

os.chdir(os.getcwd()+'/..')

import src.UK as UK
import src.data as data
import src.disp as disp


def summary_plot(x, Fc_phys, h):
    Nt = x.shape[0]
    # Time/Frequency vectors and FFT computations
    t       = np.arange(0, h*Nt, h) 
    X       = np.fft.rfft(x, axis=0)
    Xabs    = np.abs(X)
    f       = np.fft.rfftfreq(Nt, h)
    
    #
    plt.figure(figsize=(15,10))
    grid = plt.GridSpec(4, 4, wspace=0.4, hspace=0.6)
    
    plt.subplot(grid[0, :2])
    plt.plot(t, x[:,2], color="black")
    plt.plot(t, x[:,1], color="red")
    plt.plot(t, x[:,0], color="blue")
    plt.xlabel("Time [s]")
    plt.ylabel(r"$Y_S$(t) [m]")
    plt.title("STRING RESPONSE")
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-3,-3))
    
    plt.subplot(grid[1, :2])
    idx_plot = t<1e-1 
    plt.plot(t[idx_plot], x[idx_plot,2], color="black")
    plt.plot(t[idx_plot], x[idx_plot,1], color="red")
    plt.plot(t[idx_plot], x[idx_plot,0], color="blue")
    plt.xlabel("Time [s]")
    plt.ylabel(r"$Y_S$(t) [m]")
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-3,-3))
    
    plt.subplot(grid[0:2, 2:])
    idx_plot = f<1e3
    plt.semilogy(f[idx_plot], Xabs[idx_plot,2], color="black", label="Bridge")
    plt.semilogy(f[idx_plot], Xabs[idx_plot,1], color="red", label="Excitation")
    plt.semilogy(f[idx_plot], Xabs[idx_plot,0], color="blue", label="Finger")
    plt.legend()
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"$|Y_S$(f)|")
    plt.title("STRING RESPONSE")
    
    plt.subplot(grid[2, :2]) 
    plt.plot(t, Fc_phys[:,1], color="black")
    plt.xlabel("Time [s]")
    plt.ylabel(r"$F_{BS}$(t) [N]")
    plt.title("COUPLING FORCE OF THE BODY ON THE STRING")
    
    plt.subplot(grid[3, :2])
    idx_plot = t<1e-1 
    plt.plot(t[idx_plot], Fc_phys[idx_plot,1], color="black")
    plt.xlabel("Time [s]")
    plt.ylabel(r"$F_{BS}$(t) [N]")
    
    plt.subplot(grid[2, 2:])
    plt.plot(t, Fc_phys[:,0], color="blue")
    plt.xlabel("Time [s]")
    plt.ylabel(r"$F_{F}$(t) [N]")
    plt.title("COUPLING FORCE OF THE FINGER ON THE STRING")
    
    plt.subplot(grid[3, 2:])
    idx_plot = t<1e-1 
    plt.plot(t[idx_plot], Fc_phys[idx_plot,0], color="blue")
    plt.xlabel("Time [s]")
    plt.ylabel(r"$F_{F}$(t) [N]")
    
    plt.show()
    
    
n = 5

M, K, C, A, b, W, V, Wc, Vc, phi, phi_c, Fext, Fext_phys, x, xd ,xdd, Fc,\
    Fc_phys, h = data.load_simulation(n)

t = np.arange(0, h*x.shape[0], h)
    
disp.set_gui_qt()
summary_plot(x, Fc_phys, h)