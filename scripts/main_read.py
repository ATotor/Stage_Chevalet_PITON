# -*- coding: utf-8 -*-

import src.UK as UK
import src.data as data
import src.disp as disp

import matplotlib as plt
import numpy as np

n = 1

M, K, C, A, b, W, V, Wc, Vc, phi, phi_c, Fext, x, xd ,xdd, Fc, h = \
    data.load_simulation(n)

t = x.shape[0]

plt.figure(figsize=(15,5))
plt.plot(t, x[:,1])