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
t       = np.arange(0,0.1, h)
#t       = np.arange(0,10, h)
Nt = t.size

# STRING MODEL ----------------------------------------------------------------

L_s = 0.65

x_s   = np.array([0.9*L_s, L_s])  
Nx_s  = x_s.size

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

F_idx   = 0  
ts      = 0.
te      = 1e-2
Fs      = 0.
Fe      = 5.
params_s  = (t, ts, te, Fs, Fe)

Fext_s, Fext_phys_s, info_fs = UK.UK_apply_force(Nt, Nx_s, phi_s, F_idx, 
                                                 UK.UK_ramp_force, params_s)

#     # BRIDGE MODEL -----------------------------------------------------
    
L_b     = 6e-2
Nx_b    = 10
x_b     = np.linspace(0, L_b, Nx_b)

w_b = 6e-2
h_b = 0.5e-2
I_b = utils.give_I_rectangle(w_b, h_b)

params_b          = {}
params_b['Nn_b']  = 4 
params_b['L']     = L_b 
params_b['E']     = 3e9
params_b['I']     = I_b
params_b['S']     = w_b*h_b
params_b['rho']   = 800

m_b, k_b, c_b, phi_b, info_b = UK.UK_beam(x_b, params_b)

Fext_b, Fext_phys_b, info_b = UK.UK_apply_force(Nt, Nx_b, phi_b)

# BOARD MODEL -----------------------------------------------------------------

h_p     = 2e-3
a_p     = 0.5 
b_p     = 0.3
E_p     = 0.5e9
I_x_p   = utils.give_I_rectangle(b_p, h_p) 
I_y_p   = utils.give_I_rectangle(a_p, h_p)

x_p_bridge  = 0.25*a_p
x_p         = np.array([0, x_p_bridge, a_p])
y_p         = x_b + b_p/2 - L_b/2
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

m_tuple         = (m_s, m_b, m_p) 
k_tuple         = (k_s, k_b, k_p)
c_tuple         = (c_s, c_b, c_p)
Fext_tuple      = (Fext_s, Fext_b, Fext_p)
Fext_phys_tuple = (Fext_phys_s, Fext_phys_b, Fext_phys_p)
phi_tuple       = (phi_s, phi_b, phi_p)

M, K, C, phi, Fext, Fext_phys = \
    UK.UK_give_overall_model(m_tuple, k_tuple, c_tuple, Fext_tuple, 
                          Fext_phys_tuple, phi_tuple)

# CONSTRAINTS -----------------------------------------------------------------

idx_b_middle    = Nx_b//2 + Nx_b%2
idx_p_bridge    = [idx for idx in range(1,Nx_b+1)]
idx_b_bridge    = [idx for idx in range(Nx_b)]

constraints = (UK.UK_constraint_fixed(0, 0), 
                UK.UK_constraint_contact((0, 1), (1, idx_b_middle)),
                UK.UK_constraint_surface_contact((1,2),idx_b_bridge,
                                                  idx_p_bridge))

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



