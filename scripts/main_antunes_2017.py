# -*- coding: utf-8 -*-

'''
Description :   gives the UK modal model of a one-string guitare. This model 
                is from J.ANTUNES and V.DEBUT, 2017.

Variables :   M           : (Nn,Nn)       modal mass matrix
              K           : (Nn,Nn)       modal rigidity matrix
              C           : (Nn,Nn)       modal damping matrix
              W           : (Nn,Nn)       modal W matrix (such that 
                                          qdd = W @ qudd + V)
              V           : (Nn)          modal V matrix (such that 
                                          qdd = W @ qudd + V)
              Wc          : (Nn,Nn)       modal Wc matrix (such that 
                                          Fc = Wc @ qudd + Vc)
              Vc          : (Nn)          modal V matrix (such that 
                                          Fc = Wc @ qudd + Vc)
              phi         : (Nx,Nn)       string's modeshapes  
              phi_c       : (Nm,Nn)       system's modeshapes at constraints 
              Fext        : (Nt,Nn)       external modal force matrix
              q0          : (Nn)          intial modal response 
              qd0         : (Nn)          intial modal response's velocity
              qdd0        : (Nn)          intial modal response's acceleration
              Fc          : (Nn)          initial constraint modal force matrix
              h           : ()            simulation time step
'''

# LIBRARIES -------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import sys
import os

os.chdir(os.getcwd()+'/..')

import src.UK as UK
import src.data as data
import src.disp as disp

print('')

coupled = True

if coupled:
    print('Model used : ANTUNES_2017 coupled')
else:
    print('Model used : ANTUNES_2017 uncoupled')
    
# TIME ARRAY ------------------------------------------------------------------

h       = 1e-5
#t       = np.arange(0,10, h) 
t       = np.arange(0,1e-1, h) 
Nt = t.size

# STRING MODEL ----------------------------------------------------------------

L = 0.65

x   = np.array([0.33*L, 0.9*L, L])  
Nx  = x.size

params          = {}
params['Nn_s']  = 200
params['T']     = 73.9
params['L']     = L 
params['rho_l'] = 3.61e-3
params['B']     = 4e-5
params['etaF']  = 7e-5
params['etaA']  = 0.9
params['etaB']  = 2.5e-2

m_s, k_s, c_s, phi_s = UK.UK_elastic_string(x, params)

F_idx   = 1  
ts      = 0.
te      = 1e-2
Fs      = 0.
Fe      = 5.
params  = (t, ts, te, Fs, Fe)

Fext_s, Fext_phys_s = UK.UK_apply_force(Nt, Nx, phi_s, F_idx, UK.UK_ramp_force, 
                                        params)

# BOARD MODEL -----------------------------------------------------------------

params = {}

params['f_b']     = np.array([78.3, 100.2, 187.3, 207.8, 250.9, 291.8, 
                              314.7, 344.5, 399.0, 429.6, 482.9, 504.2, 
                              553.9, 580.3, 645.7, 723.5])

params['zeta_b']  = np.array([2.2, 1.1, 1.6, 1.0, 0.7, 0.9, 1.1, 0.7, 1.4, 
                              0.9, 0.7, 0.7, 0.6, 1.4, 1.0, 1.3])

params['m_b']     = np.array([2.91, 0.45, 0.09, 0.25, 2.65, 9.88, 8.75, 
                              8.80, 0.9, 0.41, 0.38, 1.07, 2.33, 1.36, 
                              2.02, 0.45])

m_b, k_b, c_b, phi_b = UK.UK_board_modal(params)

Fext_b, Fext_phys_b  = UK.UK_apply_force(Nt, 1, phi_b)

# OVERALL MODEL ---------------------------------------------------------------

m_tuple         = (m_s, m_b) 
k_tuple         = (k_s, k_b)
c_tuple         = (c_s, c_b)
Fext_tuple      = (Fext_s, Fext_b)
Fext_phys_tuple = (Fext_phys_s, Fext_phys_b)
phi_tuple       = (phi_s, phi_b)

M, K, C, phi, Fext, Fext_phys = \
    UK.UK_give_overall_model(m_tuple, k_tuple, c_tuple, Fext_tuple, 
                          Fext_phys_tuple, phi_tuple)

# CONSTRAINTS -----------------------------------------------------------------

if coupled:
    constraints = (UK.UK_constraint_fixed(0, 0), 
                   UK.UK_constraint_contact((0,1), (2,0)))
else:
    constraints = (UK.UK_constraint_fixed(0, 0), 
                   UK.UK_constraint_fixed(0, 2))

A, b, phi_c = UK.UK_give_A_b(phi_tuple, constraints)
    
# W, V, Wc, Vc MATRICES -------------------------------------------------------

W, V, Wc, Vc = UK.UK_give_W_V(A, M, b)

# INITIAL CONDITIONS ----------------------------------------------------------

initial = (UK.UK_initial_rest(0), UK.UK_initial_rest(1))

q, qd, qdd, Fc = UK.UK_give_initial_state(phi_tuple, initial)


Nt, Nn  = Fext.shape
Nc      = phi_c.shape[0]
Nx      = phi.shape[0]

# MAIN SIMULATION -------------------------------------------------------------

# Data arrays' initialization
x, xd, xdd, Fc_phys             = np.zeros((Nt, Nx)), np.zeros((Nt, Nx)), \
                                    np.zeros((Nt, Nx)), np.zeros((Nt, Nc))
x[0], xd[0], xdd[0], Fc_phys[0] = UK.give_phys(phi, phi_c, q, qd, qdd, Fc)

# Main simulation loop
print('------- Simulation running -------')
for i in range(1,Nt):
    if not (100*i/Nt)%5:
        sys.stdout.write(f"\rProgression : {100*i//Nt} %")
        sys.stdout.flush()
    q, qd, qdd, Fc                  = UK.UK_step(M, K, C, 
                                              Fext[[i]], W, 
                                              V, Wc, Vc, q, qd, qdd, h)
    x[i], xd[i], xdd[i], Fc_phys[i] = UK.give_phys(phi, phi_c, q, qd, qdd, Fc)
print('\n------- Simulation over -------')

# SAVE SIMULATION -------------------------------------------------------------

data.save_simulation(M, K, C, A, b, W, V, Wc, Vc, phi, phi_c,
                     Fext, Fext_phys, x, xd ,xdd, Fc, Fc_phys, h)
