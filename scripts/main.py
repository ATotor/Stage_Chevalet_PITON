# -*- coding: utf-8 -*-


import src.UK as UK
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

M, K, C, W, V, phi, Fext, q0, qd0 ,qdd0, h = UK.ANTUNES_2017()

Nt, Nn  = Fext.shape

q, qd, qdd  = np.zeros((Nt, Nn)), np.zeros((Nt, Nn)), \
            np.zeros((Nt, Nn))
q[0], qd[0], qdd[0] = q0, qd0, qdd0

print('------- Simulation running -------')
for i in range(1,Nt):
    if not (100*i/Nt)%5:
        print(f'{100*i//Nt} %')
    q[i], qd[i], qdd[i] = UK.UK_step(M, K, C, Fext[i], W, V, q[i-1], 
                                  qd[i-1], qdd[i-1], h)
print('------- Simulation over -------')

x, xd, xdd = UK.give_x(phi, q, qd, qdd)

for i in range(x.shape[-1]):
    plt.figure(figsize=(15,5))
    plt.plot(x[:,0])