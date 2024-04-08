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

h       = 1e-5
#t       = np.arange(0,10, h)
t       = np.arange(0,0.1, h)
Nt = t.size

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

m_s, k_s, c_s, phi_s, info_s = UK.UK_elastic_string(x_s, params_s)

F_idx       = np.argmin(np.abs(x_s - 0.9*L_s))
ts          = 0.
te          = 1e-2
Fs          = 0.
Fe          = 5.
params_s    = (t, ts, te, Fs, Fe)

Fext_s, Fext_phys_s, info_fs = UK.UK_apply_force(Nt, Nx_s, phi_s, F_idx, 
                                                 UK.UK_ramp_force, params_s)

# BOARD MODEL -----------------------------------------------------------------

h_p     = 2e-3
a_p     = 0.5 
b_p     = 0.3
E_p     = 0.5e9
I_x_p   = utils.give_I_rectangle(b_p, h_p) 
I_y_p   = utils.give_I_rectangle(a_p, h_p)

x_p_bridge  = 0.25*a_p
x_p         = np.array([0, x_p_bridge, a_p])
y_p         = [b_p/2]
y_p         = np.concatenate(([0],y_p, [b_p]))

x_p, y_p    = utils.make_grid(x_p, y_p)
Nx_p        = x_p.size 

params_p = {
    'Nm_p'  : 12, 
    'Nn_p'  : 12,
    'h'     : h_p, 
    'a'     : a_p,
    'b'     : b_p,
    'rho'   : 600,
    'D'     : (E_p*I_x_p, E_p*I_y_p, E_p*I_x_p/2, E_p*I_y_p/2),
    }

m_p, k_p, c_p, phi_p, info_p = UK.UK_board(x_p, y_p, params_p)

Fext_p, Fext_phys_p, info_fp = UK.UK_apply_force(Nt, Nx_p, phi_p)

# OVERALL MODEL ---------------------------------------------------------------

m_tuple         = (m_s, m_p) 
k_tuple         = (k_s, k_p)
c_tuple         = (c_s, c_p)
Fext_tuple      = (Fext_s, Fext_p)
Fext_phys_tuple = (Fext_phys_s, Fext_phys_p)
phi_tuple       = (phi_s, phi_p)

M, K, C, phi, Fext, Fext_phys = \
    UK.UK_give_overall_model(m_tuple, k_tuple, c_tuple, Fext_tuple, 
                          Fext_phys_tuple, phi_tuple)

# CONSTRAINTS -----------------------------------------------------------------

constraints = (UK.UK_constraint_contact((0, 1), (idx_b_s, 1)),)

A, b, phi_c, info_c = UK.UK_give_A_b(phi_tuple, constraints)

# W, V, Wc, Vc MATRICES -------------------------------------------------------

W, V, Wc, Vc = UK.UK_give_W_V(A, M, b)

# INITIAL CONDITIONS ----------------------------------------------------------

initial = (UK.UK_initial_rest(0), UK.UK_initial_rest(1))

q, qd, qdd, Fc, info_i = UK.UK_give_initial_state(phi_tuple, initial)

info = [info_s, info_p, info_fs, info_fp] + info_c + info_i

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
    x[i], xd[i], xdd[i], Fc_phys[i] = UK.give_phys(phi, phi_c, q, qd, qdd, Fc)
print('\n------- Simulation over -------')

# SAVE SIMULATION -------------------------------------------------------------

data.save_simulation(info, M, K, C, A, b, W, V, Wc, Vc, phi, phi_c,
                     Fext, Fext_phys, x, xd ,xdd, Fc, Fc_phys, h)


