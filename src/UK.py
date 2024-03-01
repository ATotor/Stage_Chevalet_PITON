# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp



'''
-------------------------------------------------------------
                   UK-SIMULATION FUNCTIONS
-------------------------------------------------------------
'''

'''
Description :   make a UK-step for a given UK model, with given modal states 
                and force.
 
Inputs :  M           : (Nn,Nn)       modal mass matrix
          K           : (Nn,Nn)       modal rigidity matrix
          C           : (Nn,Nn)       modal damping matrix
          Fext        : (Nn)          external modal force vector
          W           : (Nn,Nn)       modal W matrix (such that 
                                      qdd = W @ qudd + V)
          V           : (Nn)          modal V matrix (such that 
                                      qdd = W @ qudd + V)
          Wc          : (Nn,Nn)       modal Wc matrix (such that 
                                      Fc = Wc @ qudd + Vc)
          Vc          : (Nn)          modal V matrix (such that 
                                      Fc = Wc @ qudd + Vc)
          q_prev      : (Nn)          previous modal response 
          qd_prev     : (Nn)          previous modal response's velocity
          qdd_prev    : (Nn)          previous modal response's acceleration
          h           : ()            simulation time step
Outputs : q0          : (Nn)          current modal response 
          qd0         : (Nn)          current modal response's velocity
          qdd0        : (Nn)          current modal response's acceleration
          Fc          : (Nt,Nn)       constraint modal force matrix
'''
def UK_step(M, K, C, Fext, W, V, Wc, Vc, q_prev, qd_prev, qdd_prev, h):
    
    # Computation of the current modal response and derivative approximates 
    # thanks to the previous state
    q       = q_prev + h*qd_prev + 0.5*h**2*qdd_prev 
    qd_mid  = qd_prev + 0.5*h*qdd_prev
    
    # Computation of the current modal acceleration thanks to the UK
    # formulation
    F       = Fext - C @ qd_mid - K @ q
    F       = F[0,:]        # Flatten the F array (due to sp/np compatibility)
    qudd    = M.power(-1) @ F
    qdd     = W @ qudd + V
    qd      = qd_prev + 0.5*h*(qdd_prev + qdd)
    Fc      = Wc @ qudd + Vc
    
    return q, qd, qdd, Fc

'''
Description :   gives the physical state x, xd, xdd and constraint force 
                Fc_phys for the given modal state q, qd, qdd and modal 
                constraint force Fc.

Inputs :  phi         : (Nx,Nn)       string's modeshapes  
          phi_c       : (Nm,Nn)       system's modeshapes at constraints' 
                                      locations
          q           : ([Nt],Nn)     modal response 
          qd          : ([Nt],Nn)     modal response's velocity
          qdd         : ([Nt],Nn)     modal response's acceleration
          Fc          : ((Nt],Nn)     constraint modal force matrix
Outputs : x           : ([Nt],Nx)     physical response
          xd          : ([Nt],Nx)     physical response's velocity
          xdd         : ([Nt],Nx)     physical response's acceleration
          Fc_phys     : ([Nt],Nc)     constraint force matrix
'''
def give_phys(phi, phi_c, q, qd, qdd, Fc):
    x       = q @ phi.T
    xd      = qd @ phi.T
    xdd     = qdd @ phi.T
    Fc_phys = Fc @ give_MP_inv(phi_c)
    
    return x, xd, xdd, Fc_phys


'''
Description :   gives the Moore-Penrose right pseudoinverse of a given matrix
                M.

Inputs : M  : 2d numpy array
'''
def give_MP_inv(M):
    return M.T @ np.linalg.inv(M @ M.T)


'''
Description :   gives the W and V matrices from the UK modal formalism for a 
                given model. W and V are such that qdd = W @ qudd + V.
    
Inputs :  A           : ([Na],Nm, Nn) modal constraint matrix
          M           : (Nn,Nn)       modal mass matrix
          b           : ([Na],Nm)     modal constraint vector
Outputs : W           : ([Na],Nn,Nn)  modal W matrix (such that 
                                      qdd = W @ qudd + V)
          V           : ([Na],Nn)     modal V matrix (such that 
                                      qdd = W @ qudd + V)
'''
def give_W_V(A, M, b):
    # Case where Bp is computed off-line
    if len(A.shape)>2:
        Na,Nm,Nn = A.shape
        I           = np.eye(Nn)
        M_sq_inv    = M.power(-1/2)
        W           = np.zeros((Na, Nn, Nn))
        V           = np.zeros((Nn,1))
        for i in range(Na):
            B       = A[i] @ M_sq_inv
            Bp      = give_MP_inv(B)
            MB      = M_sq_inv @ Bp
            W[i]    = I - MB @ A[i]
            V[i]    = MB @ b[i]
    # Case where Bp is computed on-line
    else:
        M_sq_inv    = M.power(-1/2)
        B           = A @ M_sq_inv 
        Bp          = B.T @ np.linalg.inv((B @ B.T))
        MB          = M_sq_inv @ Bp
        W           = np.eye(M.shape[0]) - MB @ A
        V           = MB @ b
        
    return W, V


'''
Description :   gives UK's model external physical force matrix Fext_phys for 
                a given modal force matrix Fext and modal deformation matrix.

Inputs  : Fext_phys   : ([Nt],Nx)     external force vector 
          phi         : (Nx,Nn)       modal deformation matrix
Outputs : Fext        : ([Nt],Nn)     modal external force vector
'''
def give_Fext(Fext_phys, phi):
    return sp.sparse.csr_array(Fext_phys @ phi)

'''
-------------------------------------------------------------
                   UK-MODEL FUNCTIONS
-------------------------------------------------------------
'''

'''
Description :   gives the UK modal model of a one-string guitare. This model 
                is from J.ANTUNES and V.DEBUT, 2017.

Outputs : M           : (Nn,Nn)       modal mass matrix
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
          phi_c       : (Nm,Nn)       system's modeshapes at constraints' 
          Fext        : (Nt,Nn)       external modal force matrix
          q0          : (Nn)          intial modal response 
          qd0         : (Nn)          intial modal response's velocity
          qdd0        : (Nn)          intial modal response's acceleration
          Fc          : (Nn)          initial constraint modal force matrix
          h           : ()            simulation time step
'''
def ANTUNES_2017(coupled=True):
    if coupled:
        print('Model used : ANTUNES_2017 coupled')
    else:
        print('Model used : ANTUNES_2017 uncoupled')
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
    
    # Fext_phys_s             =  np.zeros((Nt, Nx))  # CHANGE TO SCIPY ARRAY !!!!!!!!!!!!!!!!!!!
    Fext_phys_s             = sp.sparse.csr_array((Nt,Nx))
    Fext_phys_s[t<tE,idxE]  =  5*t[t<tE]/t[t<tE][-1]
    Fext_s                  = give_Fext(Fext_phys_s, phi_s) 
    
    
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
    
    #Fext_b  = np.zeros((Nt,Nn_b)) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Fext_b   = sp.sparse.csr_array((Nt,Nn_b))
    
    # Overall model
    
    m   = np.concatenate((m_s, m_b)) 
    k   = np.concatenate((k_s, k_b))
    c   = np.concatenate((c_s, c_b))
    
    M   = sp.sparse.dia_array(np.diag(m))
    K   = sp.sparse.dia_array(np.diag(k))
    C   = sp.sparse.dia_array(np.diag(c))
    
    Nn = m.shape[0]
    
    #Fext = np.concatenate((Fext_s,Fext_b), axis=-1)
    Fext = sp.sparse.hstack([Fext_s, Fext_b])
    Fext = sp.sparse.csr_array(Fext)
    
    phi             = np.zeros((Nx + 1, Nn))
    phi[:Nx, :Nn_s] = phi_s
    phi[Nx:, Nn_s:] = phi_b
    
    phi_c           = phi[[0,2]]
    
    # Constraints
    
    A = np.zeros((2,Nn))
    A[0,:Nn_s] = phi_s[-1]
    A[1,:Nn_s] = phi_s[idxF]
    if coupled:
        A[0, Nn_s:] = -phi_b[0].T
    
    b = np.zeros((2)) 
    
    # Bp matrix
    
    M_sq        = M.power(1/2)
    M_sq_inv    = M.power(-1/2)
    B           = A @ M_sq_inv
    Bp          = B.T @ np.linalg.inv((B @ B.T))
    MB          = M_sq_inv @ Bp
    W           = np.eye(Nn) - MB @ A
    V           = MB @ b 
    Wc          = -M_sq @ Bp @ A
    Vc          = M_sq @ Bp @ b
    
    # Initial conditions
    
    q0      = np.zeros(Nn)
    qd0     = np.zeros(Nn)
    qdd0    = np.zeros(Nn) 
    Fc0     = np.zeros(Nn)
    
    return M, K, C, W, V, Wc, Vc, phi, phi_c, Fext, q0, qd0 ,qdd0, Fc0, h
    


'''
-------------------------------------------------------------
                   MAIN FUNCTION
-------------------------------------------------------------
'''

def main() -> None:
    
    import disp

    # Getting system's model and initial variables 
    M, K, C, W, V, Wc, Vc, phi, phi_c, Fext, q, qd ,qdd, Fc, h = \
        ANTUNES_2017(True)
    
    Nt, Nn  = Fext.shape
    Nc      = phi_c.shape[0]
    Nx      = phi.shape[0]
    
    # Data arrays' initialization
    x, xd, xdd, Fc_phys             = np.zeros((Nt, Nx)), np.zeros((Nt, Nx)), \
                                        np.zeros((Nt, Nx)), np.zeros((Nt, Nc))
    x[0], xd[0], xdd[0], Fc_phys[0] = give_phys(phi, phi_c, q, qd, qdd, Fc)
    
    # Main simulation loop
    print('------- Simulation running -------')
    for i in range(1,Nt):
        if not (100*i/Nt)%5:
            print(f'{100*i//Nt} %')
        q, qd, qdd, Fc                  = UK_step(M, K, C, 
                                                  Fext[[i]], W, 
                                                  V, Wc, Vc, q, qd, qdd, h)
        x[i], xd[i], xdd[i], Fc_phys[i] = give_phys(phi, phi_c, q, qd, qdd, Fc)
    print('------- Simulation over -------')
    
    # Results display
    disp.set_gui_qt()
    disp.summary_plot(x, Fc_phys, h)
    
'''
-------------------------------------------------------------
-------------------------------------------------------------
'''

if __name__ == "__main__":
     main()