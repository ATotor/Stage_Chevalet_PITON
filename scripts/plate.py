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
h       = 1e-5
#t       = np.arange(0,10, h)
t       = np.arange(0,5, h)
Nt      = t.size

# BOARD MODEL -----------------------------------------------------------------

Nm_p    = 4   
Nn_p    = 4
rho_p   = 2740
h_p     = 3.19e-3
a_p     = 0.48
b_p     = 0.42
E_p_x   = 70e9
E_p_y   = 70e9
G_p     = 0 
nu_p_xy = 0.3
nu_p_yx = nu_p_xy
# eta_p   = 0.005
# R_p     = 7
eta_p   = 0.004
R_p     = 0



x_p_impact  = 0.02
x_p         = np.arange(0,a_p,0.015)
x_p         += (a_p - x_p[-1])/2
y_p_impact  = 0.405
y_p         = np.arange(0, b_p, 0.015)
y_p         += (b_p - y_p[-1])/2

x_p, y_p    = utils.make_grid(x_p, y_p)
Nx_p        = x_p.size 

params_p = {
    'Nm_p'  : Nm_p, 
    'Nn_p'  : Nn_p,
    'h'     : h_p, 
    'a'     : a_p,
    'b'     : b_p,
    'rho'   : rho_p,
    'E_x'   : E_p_x,
    'E_y'   : E_p_y,
    'G'     : G_p,
    'nu_xy' : nu_p_xy,
    'nu_yx' : nu_p_yx,
    'eta'   : eta_p,
    'R'     : R_p
    }

m_p, k_p, c_p, phi_p, info_p = UK.UK_board(x_p, y_p, params_p)


ts          = 1e-2
te          = 3e-2
F           = 1.
F_idx       = np.argmin(utils.dist((x_p, y_p),(x_p_impact, y_p_impact)))
params_p    = (t, ts, te, F, 2500, 250)

Fext_p, Fext_phys_p, info_fp = UK.UK_apply_force(Nt, Nx_p, phi_p, F_idx, 
                                                  UK.UK_smooth_ramp_force, 
                                                  params_p)

# %%

# OVERALL MODEL ---------------------------------------------------------------

m_tuple         = (m_p,) 
k_tuple         = (k_p,)
c_tuple         = (c_p,)
Fext_tuple      = (Fext_p,)
Fext_phys_tuple = (Fext_phys_p,)
phi_tuple       = (phi_p,)

M, K, C, phi, Fext, Fext_phys = \
    UK.UK_give_overall_model(m_tuple, k_tuple, c_tuple, Fext_tuple, 
                          Fext_phys_tuple, phi_tuple)

# CONSTRAINTS -----------------------------------------------------------------

constraints = ()

A, b, phi_c, info_c = UK.UK_give_A_b(phi_tuple, constraints)

# W, V, Wc, Vc MATRICES -------------------------------------------------------

W, V, Wc, Vc = UK.UK_give_W_V(A, M, b)

# INITIAL CONDITIONS ----------------------------------------------------------

initial = (UK.UK_initial_rest(0),)

q, qd, qdd, Fc, info_i = UK.UK_give_initial_state(phi_tuple, initial)

info = [info_p, info_fp] + info_c + info_i

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
