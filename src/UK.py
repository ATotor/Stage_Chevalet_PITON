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
Description : gives a ramp-force vector

Inputs :    t       : (Nt)  time vector      
            ts      : ()    ramp's starting time
            te      : ()    ramp's ending time
            Fs      : ()    ramp's force at starting time
            Fe      : ()    ramp's force at ending time

Outputs :   F_ramp  : (Nt)  ramp-force vector
'''
def UK_ramp_force(t, ts, te, Fs, Fe):
    Nt                  = t.shape
    F_ramp              = np.zeros(Nt)
    idx_ramp            = np.logical_and(t>ts, t<te)
    F_ramp[idx_ramp]    = Fs + (Fe-Fs)*(t[idx_ramp] - ts)/(te - ts)
    return F_ramp
    

'''
Description : gives a modal external force matrix

Inputs :    Nt      : ()        time array's length
            Nx      : ()        position array's length
            F_idx   : ()        index in the position array where the force is 
                                applied 
            F_fun   : ()        force function
            phi     : ()        subsystem's modeshape
            params  : ()        force function's parameters

Outputs :   Fext    : (Nt,Nx)  modal external force vector
'''
def UK_apply_force(Nt, Nx, F_idx=0, F_fun=None, phi=None, params=()):
    Fext = sp.sparse.csr_array((Nt,Nx))
    if F_fun:
        Fext[:, F_idx]  = F_fun(*params)
        Fext            = give_Fext(Fext, phi)
    return Fext


'''
Description :   gives the UK modal parameters m, c and k and modeshapes phi_s
                for a damped elastic string 

Inputs :    x       : []        position vector    
            params  :           dictionnary containing the following 
                                parameters :
                Nn_s    : [-]       number of string modes taken into account            
                T       : [N]       string's tension force 
                L       : [m]       string's length
                rho_l   : [kg/m3]   string's longitudinal density
                B       : [Nm2]     string's bending stiffness
                etaF    : []        string's internal friction
                etaA    : []        string's air viscous damping
                etaB    : []        string's bending damping

Outputs :   m_s     : (Nn_s)        string's modal mass vector
            k_s     : (Nn_s)        string's modal rigidity vector
            c_s     : (Nn_s)        string's modal damping vector
            phi_s   : (Nx, Nn_s)    string's modeshapes 
'''
def UK_elastic_string(x, params):
    Nn_s    = params['Nn_s']
    T       = params['T']
    L       = params['L']
    rho_l   = params['rho_l']
    B       = params['B']
    etaF    = params['etaF']
    etaA    = params['etaA']
    etaB    = params['etaB']
    n       = np.arange(Nn_s) + 1
    ct      = np.sqrt(T/rho_l)
    
    Nx = x.size
    
    p = (2*n-1)*np.pi/(2*L)
    f_s = ct/(2*np.pi)*p*(1+B/(2*T)*p**2)    
    w_s = 2*np.pi*f_s
    
    phi_s = np.zeros((Nx, Nn_s))
    for i in range(n.size):
        phi_s[:,i] = np.sin(p[i]*x)
    
    m_s = rho_l*L/2*np.ones(Nn_s)
    
    zeta_s = 1/2*(T*(etaF+etaA/w_s)+etaB*B*p**2)/(T+B*p**2) 
    
    k_s = m_s*(w_s)**2
    
    c_s = 2*m_s*w_s*zeta_s
    
    return m_s, k_s, c_s, phi_s


'''
Description :   gives the UK modal parameters m, c and k and modeshapes phi_s
                for a modal board

Inputs :    params  :           dictionnary containing the following 
                                parameters :
                f_b     : (Nn_b) boards' modal frequencies
                zeta_b  : (Nn_b) boards' modal losses
                m_b     : (Nn_b) boards' modal masses

Outputs :   m_b     : (Nn_s)        board's modal mass vector
            k_b     : (Nn_s)        board's modal rigidity vector
            c_b     : (Nn_s)        board's modal damping vector
            phi_b   : (Nx, Nn_s)    board's modeshapes 
'''
def UK_board_modal(params):
    f_b     = params['f_b']
    zeta_b  = params['zeta_b']
    m_b     = params['m_b']
    
    Nn_b    = f_b.size 
    w_b     = 2*np.pi*f_b
    k_b     = m_b*(w_b)**2
    c_b     = 2*m_b*w_b*zeta_b
    phi_b   = np.ones((1,Nn_b))  # body modes normalized at the bridge location
    
    return m_b, k_b, c_b, phi_b



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
          phi_c       : (Nm,Nn)       system's modeshapes at constraints 
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
    
    L = 0.65
    
    x       = np.array([0.33*L, 0.9*L, L])  
    Nx      = x.size
    params  = {}
    params['Nn_s']  = 200
    params['T']     = 73.9
    params['L']     = L 
    params['rho_l'] = 3.61e-3
    params['B']     = 4e-5
    params['etaF']  = 7e-5
    params['etaA']  = 0.9
    params['etaB']  = 2.5e-2
    
    m_s, k_s, c_s, phi_s = UK_elastic_string(x, params)
    
    F_idx   = 1  
    ts      = 0.
    te      = 1e-2
    Fs      = 0.
    Fe      = 5.
    params  = (t, ts, te, Fs, Fe)
    
    Fext_s = UK_apply_force(Nt, Nx, F_idx, UK_ramp_force, phi_s, params)
    
    # Board model
    
    params = {}
    
    params['f_b']     = np.array([78.3, 100.2, 187.3, 207.8, 250.9, 291.8, 
                                  314.7, 344.5, 399.0, 429.6, 482.9, 504.2, 
                                  553.9, 580.3, 645.7, 723.5])
    
    params['zeta_b']  = np.array([2.2, 1.1, 1.6, 1.0, 0.7, 0.9, 1.1, 0.7, 1.4, 
                                  0.9, 0.7, 0.7, 0.6, 1.4, 1.0, 1.3])
    
    params['m_b']     = np.array([2.91, 0.45, 0.09, 0.25, 2.65, 9.88, 8.75, 
                                  8.80, 0.9, 0.41, 0.38, 1.07, 2.33, 1.36, 
                                  2.02, 0.45])
    
    m_b, k_b, c_b, phi_b = UK_board_modal(params)
    
    Fext_b  = UK_apply_force(Nt, 1)
    
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