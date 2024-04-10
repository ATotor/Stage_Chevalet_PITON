# -*- coding: utf-8 -*-

# LIBRARIES -------------------------------------------------------------------

import numpy as np
import scipy as sp
import sys
import os

os.chdir(os.getcwd()+'/..')

import src.utils as utils
import src.data as data
import src.UK as UK

# TIME ARRAY ------------------------------------------------------------------

#h       = 1e-5
h       = 2.5e-6
#t       = np.arange(0,10, h)
t       = np.arange(0,0.2, h)
Nt      = t.size

# STRING MODEL ----------------------------------------------------------------

L_s = 0.65

Nx_s    = 100 
x_s     = np.linspace(0, L_s, Nx_s)
#x_s     = np.array([0, 0.9*L_s, L_s])
idx_b_s = Nx_s - 1
Nx_s    = x_s.size

params_s          = {}
params_s['Nn_s']  = 200
params_s['T']     = 73.9
params_s['L']     = L_s 
params_s['rho_l'] = 3.61e-3
params_s['B']     = 0.
params_s['etaF']  = 0.
params_s['etaA']  = 0.
params_s['etaB']  = 0.
# params_s['B']     = 4e-5
# params_s['etaF']  = 7e-5
# params_s['etaA']  = 0.9
# params_s['etaB']  = 2.5e-2

m_s, k_s, c_s, phi_s, info_s = UK.UK_elastic_string(x_s, params_s)

F_idx       = np.argmin(np.abs(x_s - 0.9*L_s))

# ts          = 0.
# te          = 1e-2
# Fs          = 0.
# Fe          = 5.
# params_s    = (t, ts, te, Fs, Fe)
# Fext_s, Fext_phys_s, info_fs = UK.UK_apply_force(Nt, Nx_s, phi_s, F_idx, 
#                                                   UK.UK_ramp_force, params_s)

ts          = 1e-2
te          = 2e-2
F           = 5.
params_s    = (t, ts, te, F, 10000, 10000)
Fext_s, Fext_phys_s, info_fs = UK.UK_apply_force(Nt, Nx_s, phi_s, F_idx, 
                                                  UK.UK_smooth_ramp_force, 
                                                  params_s)

# mu  = 0.01
# sig = 0.0025
# F   = 5
# params_s = (t, mu, sig, F)
# Fext_s, Fext_phys_s, info_fs = UK.UK_apply_force(Nt, Nx_s, phi_s, F_idx, 
#                                                   UK.UK_force_gaussian, params_s)

# OVERALL MODEL ---------------------------------------------------------------

m_tuple         = (m_s,) 
k_tuple         = (k_s,)
c_tuple         = (c_s,)
Fext_tuple      = (Fext_s,)
Fext_phys_tuple = (Fext_phys_s,)
phi_tuple       = (phi_s,)

M, K, C, phi, Fext, Fext_phys = \
    UK.UK_give_overall_model(m_tuple, k_tuple, c_tuple, Fext_tuple, 
                          Fext_phys_tuple, phi_tuple)

# CONSTRAINTS -----------------------------------------------------------------

constraints = (UK.UK_constraint_fixed(0, idx_b_s),)

A, b, phi_c, info_c = UK.UK_give_A_b(phi_tuple, constraints)

# W, V, Wc, Vc MATRICES -------------------------------------------------------

W, V, Wc, Vc = UK.UK_give_W_V(A, M, b)

# INITIAL CONDITIONS ----------------------------------------------------------

initial = (UK.UK_initial_rest(0),)

q, qd, qdd, Fc, info_i = UK.UK_give_initial_state(phi_tuple, initial)

info = [info_s, info_fs] + info_c + info_i

# MAIN SIMULATION -------------------------------------------------------------

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
    if not (100*i/Nt)%5:
        sys.stdout.write(f"\rProgression : {100*i//Nt} %")
        sys.stdout.flush()
    q, qd, qdd, Fc                  = UK.UK_step(M, K, C, 
                                              Fext[[i]], W, 
                                              V, Wc, Vc, q, qd, qdd, h)
    q, qd                           = UK.violation_elim(q, qd, constraints, 
                                                        phi_c, A)
    x[i], xd[i], xdd[i], Fc_phys[i] = UK.give_phys(phi, phi_c, q, qd, qdd, Fc)
print('\n------- Simulation over -------')

# SAVE SIMULATION -------------------------------------------------------------

data.save_simulation(info, M, K, C, A, b, W, V, Wc, Vc, phi, phi_c,
                     Fext, Fext_phys, x, xd ,xdd, Fc, Fc_phys, h)
