# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os

os.chdir(os.getcwd()+'/..')

import src.UK as UK
import src.data as data
import src.disp as disp

n = 17

M, K, C, A, b, W, V, Wc, Vc, phi, phi_c, Fext, Fext_phys, x, xd ,xdd, Fc,\
    Fc_phys, h = data.load_simulation(n)

t = np.arange(0, h*x.shape[0], h)


disp.set_gui_qt()

plt.figure(figsize=(15,5))
plt.plot(t, x[:,0])

plt.figure(figsize=(15,5))
plt.plot(t, x[:,1])

plt.figure(figsize=(15,5))
plt.plot(t, Fc_phys[:])