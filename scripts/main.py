# -*- coding: utf-8 -*-

import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt

import src.UK as UK
import src.disp as disp

print('')

# Getting the user's input
if __name__ == "__main__":
     parser = argparse.ArgumentParser(description="Run a specific function.")
     parser.add_argument("--uncoupled", action = "store_true", 
                         help='Whether the string and body of the instrument' +
                         'are uncoupled')

     args = parser.parse_args()
     coupled = not args.uncoupled

# Getting system's model and initial variables 
M, K, C, W, V, Wc, Vc, phi, phi_c, Fext, q, qd ,qdd, Fc, h = \
    UK.ANTUNES_2017(coupled)

Nt, Nn  = Fext.shape
Nc      = phi_c.shape[0]
Nx      = phi.shape[0]

# Data arrays' initialization
x, xd, xdd, Fc_phys             = np.zeros((Nt, Nx)), np.zeros((Nt, Nx)), \
                                    np.zeros((Nt, Nx)), np.zeros((Nt, Nc))
x[0], xd[0], xdd[0], Fc_phys[0] = UK.give_phys(phi, phi_c, q, qd, qdd, Fc)

# Main simulation loop
print('------- Simulation running -------')
for i in range(1,Nt):
    if (not (100*i/Nt)%5) or i==1 :
        sys.stdout.write(f"\rProgression : {100*i//Nt} %")
        sys.stdout.flush()
    q, qd, qdd, Fc                  = UK.UK_step(M, K, C, Fext[i], W, V, Wc, \
                                              Vc, q, qd, qdd, h)
    x[i], xd[i], xdd[i], Fc_phys[i] = UK.give_phys(phi, phi_c, q, qd, qdd, Fc)
print('\n------- Simulation over -------')

# Results display
disp.set_gui_qt()
disp.summary_plot(x, Fc_phys, h)