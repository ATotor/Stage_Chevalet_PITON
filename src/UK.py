# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp


# Inputs :  M           : (Nn,Nn)       modal mass matrix
#           K           : (Nn,Nn)       modal rigidity matrix
#           C           : (Nn,Nn)       modal damping matrix
#           Fext        : (Nn)          external modal force vector
#           W           : (Nn,Nn)       modal W matrix (such that 
#                                       qdd = W @ qudd + V)
#           V           : (Nn)          modal V matrix (such that 
#                                       qdd = W @ qudd + V)
#           q_prev      : (Nn)          previous modal response 
#           qd_prev     : (Nn)          previous modal response's velocity
#           qdd_prev    : (Nn)          previous modal response's acceleration
#           h           : ()            simulation time step
# Outputs : q0          : (Nn)          current modal response 
#           qd0         : (Nn)          current modal response's velocity
#           qdd0        : (Nn)          current modal response's acceleration
def UK_step(M, K, C, Fext, W, V, q_prev, qd_prev, qdd_prev, h):
    
    # Computation of the current modal response and derivative approximates 
    # thanks to the previous state
    q       = q_prev + h*qd_prev + 0.5*h**2*qdd_prev 
    qd_mid  = qd_prev + 0.5*h*qdd_prev
    
    # Computation of the current modal acceleration thanks to the UK
    # formulation
    F       = Fext - C@qd_mid - K@q
    qudd    = M.power(-1)@F
    qdd     = W @ qudd + V
    qd      = qd_prev + 0.5*h*(qdd_prev + qdd)
    
    return q, qd, qdd


# Inputs :  A           : (Nm, Nn)      modal constraint matrix
#           M           : (Nn,Nn)       modal mass matrix
#           b           : (Nm)          modal constraint vector
# Outputs : W           : (Nn,Nn)       modal W matrix (such that 
#                                       qdd = W @ qudd + V)
#           V           : (Nn)          modal V matrix (such that 
#                                       qdd = W @ qudd + V)
def give_W_V(A, M, b):
    # Case where Bp is computed off-line
    if len(A.shape)>2:
        Nc,Nm,Nn = A.shape
        I           = np.eye(Nn)
        M_sq_inv    = M.power(-1/2)
        W           = np.zeros((Nc, Nn, Nn))
        V           = np.zeros((Nn,1))
        for i in range(Nc):
            B       = A[i] @ M_sq_inv
            Bp      = B.T @ np.linalg.inv(B @ B.T)
            MB      = M_sq_inv @ Bp
            W[i]    = I - MB @ A[i]
            V[i]    = MB @ b
    # Case where Bp is computed on-line
    else:
        M_sq_inv    = M.power(-1/2)
        B           = A @ M_sq_inv 
        Bp          = B.T @ np.linalg.inv((B @ B.T))
        MB          = M_sq_inv @ Bp
        W           = np.eye(M.shape[0]) - MB @ A
        V           = MB @ b
        
    return W, V


# Inputs  : Fext_phys   : (Nx)         external force vector 
#           phi         : (Nx,Nn)       modal deformation vector
# Outputs : Fext        : (Nn)          modal external force vector
def give_Fext(Fext_phys, phi):
    return Fext_phys @ phi


# Outputs : M           : (Nn,Nn)       modal mass matrix
#           K           : (Nn,Nn)       modal rigidity matrix
#           C           : (Nn,Nn)       modal damping matrix
#           W           : (Nn,Nn)       modal W matrix (such that 
#                                       qdd = W @ qudd + V)
#           V           : (Nn)          modal V matrix (such that 
#                                       qdd = W @ qudd + V)
#           phi         : (Nx,Nn)      string's modeshapes  
#           Fext        : (Nt,Nn)       external modal force matrix
#           q0          : (Nn)          intial modal response 
#           qd0         : (Nn)          intial modal response's velocity
#           qdd0        : (Nn)          intial modal response's acceleration
#           h           : ()            simulation time step
def ANTUNES_2017():
    h       = 1e-5
    t       = np.arange(0,10, h) 
    tE      = 1e-2
    
    Nt = t.size
    
    # String model
    
    L       = 0.65 
    xE      = 0.9*L
    xF      = 0.33*L
    idxF    = 0
    idxE    = 1
    x       = np.array([xF, xE, L])
    n       = np.arange(200)+1
    T       = 73.9
    rho_l   = 3.61e-3
    ct      = np.sqrt(T/rho_l)
    B       = 4e-5
    etaF    = 7e-5
    etaA    = 0.9
    etaB    = 2.5e-2
    
    Nn_s    = n.size
    Nx      = x.size
    
    p = (2*n-1)*np.pi/(2*L)
    f_s = ct/(2*np.pi)*p*(1+B/(2*T)*p**2)    
    w_s = 2*np.pi*f_s
    
    phi_s = np.zeros((Nx, Nn_s))
    for i in range(n.size):
        phi_s[:,i] = np.sin(p[i]*x)
    
    Fext_phys_s             =  np.zeros((Nt, Nx))
    Fext_phys_s[t<tE,idxE]  =  5*t[t<tE]/t[t<tE][-1]
    Fext_s                  = np.zeros((Nt, Nn_s))
    for i in range(Nn_s):
        Fext_s[i] = give_Fext(Fext_phys_s[i], phi_s)  # BUG HERE !!!!!!!!!!!!!!!!!
    
    m_s = rho_l*L/2*np.ones(Nn_s)
    
    zeta_s = 1/2*(T*(etaF+etaA/w_s)+etaB*B*p**2)/(T+B*p**2) 
    
    k_s = m_s*(w_s)**2
    
    c_s = 2*m_s*w_s*zeta_s
    
    
    # Board model
    
    Nn_b    = 16 
    
    phi_b   = np.ones((1,Nn_b))  # body modes normalized at the bridge location
    
    f_b     = np.array([78.3, 100.2, 187.3, 207.8, 250.9, 291.8, 314.7, 
                        344.5, 399.0, 429.6, 482.9, 504.2, 553.9, 580.3, 
                        645.7, 723.5]) 
    w_b     = 2*np.pi*f_b
    
    zeta_b  = np.array([2.2, 1.1, 1.6, 1.0, 0.7, 0.9, 1.1, 0.7, 1.4, 0.9, 
                        0.7, 0.7, 0.6, 1.4, 1.0, 1.3])
    
    m_b     = np.array([2.91, 0.45, 0.09, 0.25, 2.65, 9.88, 8.75, 8.80, 0.9, 
                        0.41, 0.38, 1.07, 2.33, 1.36, 2.02, 0.45])
    
    k_b = m_b*(w_b)**2
    
    c_b = 2*m_b*w_b*zeta_b
    
    Fext_b  = np.zeros((Nt,Nn_b))
    
    # Overall model
    
    m   = np.concatenate((m_s, m_b)) 
    k   = np.concatenate((k_s, k_b))
    c   = np.concatenate((c_s, c_b))
    
    M   = sp.sparse.dia_array(np.diag(m))
    K   = sp.sparse.dia_array(np.diag(k))
    C   = sp.sparse.dia_array(np.diag(c))
    
    Fext = np.concatenate((Fext_s,Fext_b), axis=-1)
    
    Nn = m.shape[0]
    
    phi = np.zeros((Nx + 1, Nn))
    phi[:Nx, :Nn_s] = phi_s
    phi[Nx:, Nn_s:] = phi_b
    
    # Constraints
    
    A = np.zeros((2,Nn))
    A[0,:Nn_s] = phi_s[-1]
    A[1,:Nn_s] = phi_s[idxF]
    
    b = np.zeros((2)) 
    
    # Bp matrix
    
    M_sq_inv    = M.power(-1/2)
    B           = A @ M_sq_inv
    Bp          = B.T @ np.linalg.inv((B @ B.T))
    MB          = M_sq_inv @ Bp
    W           = np.eye(Nn) - MB @ A
    V           = MB @ b 
    
    # Initial conditions
    
    q0      = np.zeros(M.shape[0])
    qd0     = np.zeros(M.shape[0])
    qdd0    = np.zeros(M.shape[0]) 
    
    return M, K, C, W, V, phi, Fext, q0, qd0 ,qdd0, h


# Inputs :  phi         : (Nx,Nn)       string's modeshapes  
#           q           : (Nt,Nn)       modal response 
#           qd          : (Nt,Nn)       modal response's velocity
#           qdd         : (Nt,Nn)       modal response's acceleration
# Outputs : x           : (Nt,Nx)       physical response
#           xd          : (Nt,Nx)       physical response's velocity
#           xdd         : (Nt,Nx)       physical response's acceleration
def give_x(phi, q, qd, qdd):
    return q @ phi.T, qd @ phi.T, qdd @ phi.T
    

# def main() -> None:
#     import matplotlib.pyplot as plt
    
#     M, K, C, W, V, phi, Fext, q0, qd0 ,qdd0, h = ANTUNES_2017()
    
#     Nt, Nn  = Fext.shape
    
#     q, qd, qdd  = np.zeros((Nt, Nn)), np.zeros((Nt, Nn)), \
#                 np.zeros((Nt, Nn))
#     q[0], qd[0], qdd[0] = q0, qd0, qdd0
    
#     print('------- Simulation running -------')
#     for i in range(1,Nt):
#         if not (100*i/Nt)%5:
#             print(f'{100*i//Nt} %')
#         q[i], qd[i], qdd[i] = UK_step(M, K, C, Fext[i], W, V, q[i-1], 
#                                       qd[i-1], qdd[i-1], h)
#     print('------- Simulation over -------')
    
#     x, xd, xdd = give_x(phi, q, qd, qdd)
    
#     #plt.plot(x[:,0])
#     #plt.plot(x[:,1])
#     #plt.plot(x[:,2])
#     print('hi')
    
#     return
    

# if __name__ == "__main__":
#     main()
    
import matplotlib.pyplot as plt

M, K, C, W, V, phi, Fext, q0, qd0 ,qdd0, h = ANTUNES_2017()

Nt, Nn  = Fext.shape

q, qd, qdd  = np.zeros((Nt, Nn)), np.zeros((Nt, Nn)), \
            np.zeros((Nt, Nn))
q[0], qd[0], qdd[0] = q0, qd0, qdd0

print('------- Simulation running -------')
for i in range(1,Nt):
    if not (100*i/Nt)%5:
        print(f'{100*i//Nt} %')
    q[i], qd[i], qdd[i] = UK_step(M, K, C, Fext[i], W, V, q[i-1], 
                                  qd[i-1], qdd[i-1], h)
print('------- Simulation over -------')

x, xd, xdd = give_x(phi, q, qd, qdd)

plt.figure(figsize=(15,5))
plt.plot(np.arange(0,h*4000,h), x[:4000,1])