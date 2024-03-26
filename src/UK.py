# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp

import src.utils as utils
import src.data as data



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
          Wc          : ([Na],Nn,Nn)  modal Wc matrix (such that 
                                      Fc = Wc @ qudd + Vc)
          Vc          : ([Na],Nn)     modal V matrix (such that 
                                      Fc = Wc @ qudd + Vc)
'''
def UK_give_W_V(A, M, b):
    # Case where Bp is computed off-line
    if len(A.shape)>2:
        Na,Nm,Nn = A.shape
        I           = np.eye(Nn)
        M_sq        = M.power(1/2)
        M_sq_inv    = M.power(-1/2)
        W           = np.zeros((Na, Nn, Nn))
        V           = np.zeros((Na, Nn, 1))
        Wc          = np.zeros((Na, Nn, Nn))
        Vc          = np.zeros((Na, Nn, 1))
        for i in range(Na):
            B       = A[i] @ M_sq_inv
            Bp      = give_MP_inv(B)
            MB      = M_sq_inv @ Bp
            W[i]    = I - MB @ A[i]
            V[i]    = MB @ b[i]
            Wc[i]   = -M_sq @ Bp @ A[i]
            Vc[i]   = M_sq @ Bp @ b[i]
    # Case where Bp is computed on-line
    else:
        M_sq        = M.power(1/2)
        M_sq_inv    = M.power(-1/2)
        B           = A @ M_sq_inv 
        Bp          = B.T @ np.linalg.inv((B @ B.T))
        MB          = M_sq_inv @ Bp
        W           = np.eye(M.shape[0]) - MB @ A
        V           = MB @ b
        Wc          = -M_sq @ Bp @ A
        Vc          = M_sq @ Bp @ b
        
    return W, V, Wc, Vc


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
            phi     : ()        subsystem's modeshape
            F_idx   : ()        index in the position array where the force is 
                                applied 
            F_fun   : ()        force function
            params  : ()        force function's parameters

Outputs :   Fext    : (Nt,Nx)  modal external force vector
'''
def UK_apply_force(Nt, Nx, phi, F_idx=0, F_fun=None, params=()):
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
                rho_l   : [kg/m]    string's longitudinal density
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
Description :   gives the UK modal parameters m, c and k and modeshapes phi_b
                for a beam 

Inputs :    x       : []        position vector    
            params  :           dictionnary containing the following 
                                parameters :
                Nn_s    : [-]       number of beam modes taken into account            
                E       : [N/m2]    beam's Young modulus    
                I       : [m2]      beam's second moment of area
                L       : [m]       beam's length
                rho     : [kg/m3]   beam's density

Outputs :   m_b     : (2*Nn_b)      beam's modal mass vector
            k_b     : (2*Nn_b)      beam's modal rigidity vector
            c_b     : (2*Nn_b)      beam's modal damping vector
            phi_b   : (Nx, 2*Nn_b)  beam's modeshapes 
'''
def UK_beam(x, params):
    Nn_b    = params['Nn_b']
    L       = params['L']
    E       = params['E']
    I       = params['I']
    S       = params['S']
    rho     = params['rho']
    
    n   = np.arange(Nn_b) + 1
    Nx  = x.size
    
    f_b     = np.sqrt(E*I/(rho*S))*np.pi/(8*L**2)*(2*n+1)**2
    w_b     = 2*np.pi*f_b
    k_b     = (2*n+1)*np.pi/(2*L)
    
    alpha       = np.sinh(2*k_b*L)/(4*k_b) + (-1)**n*np.cosh(k_b*L)/k_b
    m_b_1       = rho*S*(L+alpha)
    m_b_2       = rho*S*alpha
    m_b         = np.zeros(2*Nn_b)
    m_b[::2]    = m_b_1
    m_b[1::2]   = m_b_2
    
    zeta_b = np.zeros(2*Nn_b)
    
    phi_b_1         = np.cos(np.outer(x,k_b)) + np.cosh(np.outer(x,k_b))
    phi_b_2         = np.sin(np.outer(x,k_b)) + np.sinh(np.outer(x,k_b))
    phi_b           = np.zeros((Nx,2*Nn_b))
    phi_b[:,::2]    = phi_b_1 
    phi_b[:,1::2]   = phi_b_2 
    
    w_b     = np.repeat(w_b,2) 
    
    k_b     = m_b*(w_b)**2
    
    c_b     = 2*m_b*w_b*zeta_b
    
    return m_b, k_b, c_b, phi_b


'''
Description :   gives the UK modal parameters m, c and k and modeshapes phi_p
                for a thin elastic anisotropic board 

Inputs :    x       : (Nx)      position vector in the x direction
            y       : (Ny)      position vector in the y direction
            params  :           dictionnary containing the following 
                                parameters :
                Nm_p    : [-]       number of board modes taken into account 
                                    in the x direction 
                Nn_p    : [-]       number of board modes taken into account
                                    in the y direction
                h       : [m]       board's thickness
                a       : [m]       board's length in the x direction
                b       : [m]       board's length in the y direction
                rho     : [kg/m3]   board's density
                D       : [Nm2]     board's bending stiffness parameters

Outputs :   m_p     : (Nm_p + Nn_p)      board's modal mass vector
            k_p     : (Nm_p + Nn_p)      board's modal rigidity vector
            c_p     : (Nm_p + Nn_p)      board's modal damping vector
            phi_p   : (Nx, Nm_p + Nn_p)  board's modeshapes 
'''
def UK_board(x, y, params):
    Nm_p    = params['Nm_p'] 
    Nn_p    = params['Nn_p']
    h       = params['h']
    a       = params['a']
    b       = params['b']
    rho     = params['rho']
    D       = params['D']
    
    D1 = D[0] 
    D2 = D[1]
    D3 = D[2]
    D4 = D[3]
    
    m   = np.arange(Nm_p) + 1
    n   = np.arange(Nn_p) + 1
    m,n = utils.make_grid(m,n)
    
    f_p     = np.pi/2*np.sqrt(1/(rho*h))*np.sqrt(D1*m**4/a**4 + D3*n**4/b**4+\
                                             (D2+D4)*m**2*n**2/(a**2*b**2))
        
    w_p     = 2*np.pi*f_p 
    
    phi_p   = np.sin(np.inner(x[:,np.newaxis],m[:,np.newaxis])*np.pi/a)*\
                np.sin(np.inner(y[:,np.newaxis],n[:,np.newaxis])*np.pi/b)
    
    m_p     = rho*h*a*b/4*np.ones(m.size) 
    
    zeta_p  = np.zeros(m.size)
    
    k_p     = m_p*(w_p)**2
    
    c_p     = 2*m_p*w_p*zeta_p
    
    return m_p, k_p, c_p, phi_p


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
Description :   gives the constraint dictionnary for a fixed constraint on the 
                given system at a given position.

Inputs :    idx_sys : index of the constrained system
            idx_x   : index of the position 
                        
Outputs :   constraint : corresponding constraint dictionnary 
'''
def UK_constraint_fixed(idx_sys, idx_x):
    constraint = {'type' : 'fixed',
                  'idx_sys' : idx_sys,
                  'idx_x' : idx_x} 
    return constraint


'''
Description :   gives the constraint dictionnary for a contact constraint on 
                the given systems at given positions.

Inputs :    idx_sys : indices of the two constrained systems
            idx_x   : indices of the two positions 
                        
Outputs :   constraint : corresponding constraint dictionnary 
'''    
def UK_constraint_contact(idx_sys, idx_x):
    constraint = {'type' : 'contact',
                  'idx_sys' : idx_sys,
                  'idx_x' : idx_x} 
    return constraint


'''
Description :   gives the initial dictionnary for a rest initial state on 
                the given system.

Inputs :    idx_sys : indices of the resting system
                        
Outputs :   constraint : corresponding initial dictionnary 
'''    
def UK_initial_rest(idx_sys):
    initial = {'type' : 'rest',
                  'idx_sys' : idx_sys} 
    return initial


'''
Description :   gives the constraint matrix A and vector b for the given 
                constraints.

Inputs :    phi_tuple   : (Ns)  tuple containing all the modeshapes (Nx_i, 
                                                                     Nn_i)
            constraints :           tuple of dictionnaries containing the 
                                    constraint parameters :
                type : string containing the type of constraint
                    if type=='fixed':
                        idx_sys : int containing the system's index
                        idx_x   : int containing the position index
                    if type=='contact':
                        idx_sys : tuple containing the two systems' indices
                        idx_x   : tuple containing the two positions' indices
                        
Outputs :   A       : (Nm, Nn)      constraint matrix
            b       : (Nm)          constraint vector
            phi_c   : (Nm,Nn)       system's modeshapes at constraints
'''
def UK_give_A_b(phi_tuple, constraints=()):
    Nn_tuple = (0,)
    for phi in phi_tuple:
        Nn_tuple += (Nn_tuple[-1] + phi.shape[-1],) 
    Nn          = Nn_tuple[-1]
    Nm          = len(constraints)
    
    if Nm==0:
        A       = np.zeros((1,Nn))
        b       = np.zeros((1))
        phi_c   = np.zeros(Nn) 
    else:
        A       = np.zeros((Nm, Nn))
        b       = np.zeros((Nm))
        phi_c   = np.zeros((Nm,Nn))
        for i,c in enumerate(constraints):
            if c['type'] == 'contact':
                idx_sys_1, idx_sys_2                            = c['idx_sys']
                idx_x_1, idx_x_2                                = c['idx_x']
                A[i,Nn_tuple[idx_sys_1]:Nn_tuple[idx_sys_1+1]]  = \
                    phi_tuple[idx_sys_1][idx_x_1]
                A[i,Nn_tuple[idx_sys_2]:Nn_tuple[idx_sys_2+1]]  = \
                    -phi_tuple[idx_sys_2][idx_x_2]
                phi_c[i,Nn_tuple[idx_sys_1]:Nn_tuple[idx_sys_1+1]] = \
                    phi_tuple[idx_sys_1][idx_x_1]
            elif c['type'] == 'fixed':
                idx_sys                                     = c['idx_sys']
                idx_x                                       = c['idx_x']
                A[i,Nn_tuple[idx_sys]:Nn_tuple[idx_sys+1]]  = \
                    phi_tuple[idx_sys][idx_x]
                phi_c[i,Nn_tuple[idx_sys]:Nn_tuple[idx_sys+1]] = \
                    phi_tuple[idx_sys][idx_x]
    return A, b, phi_c


'''
Description :   gives the initial state q0, qd0, qdd0 and the initial 
                constraint force for the given initial conditions.

Inputs :    phi_tuple   : (Ns)  tuple containing all the modeshapes (Nx_i, 
                                                                     Nn_i)
            initials    :       tuple of dictionnaries containing the 
                                    initial conditions :
                type : string containing the type of initial conditions
                    if type=='rest':
                        idx_sys : int containing the system's index
                        
Outputs :   q0      : (Nn) initial position vector
            qd0     : (Nn) initial velocity vector
            qdd0    : (Nn) initial acceleration vector
            Fc0     : (Nn) initial constraint force vector
'''
def UK_give_initial_state(phi_tuple, initials=()):
    Nn_tuple = (0,)
    for phi in phi_tuple:
        Nn_tuple += (Nn_tuple[-1] + phi.shape[-1],) 
    Nn          = Nn_tuple[-1]
    
    q0      = np.zeros(Nn)
    qd0     = np.zeros(Nn)
    qdd0    = np.zeros(Nn)
    Fc0     = np.zeros(Nn)
    
    for i,init in enumerate(initials):
        if init['type'] == 'rest':
            idx_sys     = init['idx_sys']
            zero_vect   = np.zeros(Nn_tuple[idx_sys+1] - Nn_tuple[idx_sys])
            q0[Nn_tuple[idx_sys]:Nn_tuple[idx_sys+1]]   = zero_vect
            qd0[Nn_tuple[idx_sys]:Nn_tuple[idx_sys+1]]  = zero_vect
            qdd0[Nn_tuple[idx_sys]:Nn_tuple[idx_sys+1]] = zero_vect
            Fc0[Nn_tuple[idx_sys]:Nn_tuple[idx_sys+1]]  = zero_vect
    
    return q0, qd0, qdd0, Fc0

'''
Description :   gives the modal matrices M, K, C, modeshapes' matrix PHI and
                modal external force matrix Fext 

Inputs :    m_tuple     : (Ns)  tuple containing all the modal masses
            k_tuple     : (Ns)  tuple containing all the modal rigidities
            c_tuple     : (Ns)  tuple containing all the modal damplings
            Fext_tuple  : (Ns)  tuple containing all the modal external forces
            phi_tuple   : (Ns)  tuple containing all the modeshapes (Nx_i, 
                                                                     Nn_i)
                        
Outputs :   M       : (Nn, Nn)      modal mass matrix
            K       : (Nn, Nn)      modal rigidity matrix
            C       : (Nn, Nn)      modal damping matrix
            PHI     : (Nx, Nn)      modeshapes' matrix
            Fext    : (Nt, Nn)      modal external force matrix
'''
def UK_give_overall_model(m_tuple, k_tuple, c_tuple, Fext_tuple, phi_tuple):
    Nn_tuple = (0,)
    Nx_tuple = (0,)
    for phi in phi_tuple:
        Nn_tuple += (Nn_tuple[-1] + phi.shape[1],)
        Nx_tuple += (Nx_tuple[-1] + phi.shape[0],)
    Nn = Nn_tuple[-1]
    Nx = Nx_tuple[-1]
    
    m   = np.concatenate(m_tuple) 
    k   = np.concatenate(k_tuple)
    c   = np.concatenate(c_tuple)
    
    M   = sp.sparse.dia_array(np.diag(m))
    K   = sp.sparse.dia_array(np.diag(k))
    C   = sp.sparse.dia_array(np.diag(c))
    
    Nn = m.shape[0]
    
    Fext = sp.sparse.hstack(Fext_tuple)
    Fext = sp.sparse.csr_array(Fext)
    
    PHI             = np.zeros((Nx + 1, Nn))
    for (i,phi) in enumerate(phi_tuple):
        PHI[Nx_tuple[i]:Nx_tuple[i+1], Nn_tuple[i]:Nn_tuple[i+1]] = phi
    
    return M, K, C, PHI, Fext
                



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
        
    # TIME ARRAY -------------------------------------------------------
    
    h       = 1e-5
    t       = np.arange(0,10, h) 
    
    Nt = t.size
    
    # STRING MODEL -----------------------------------------------------
    
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
    
    m_s, k_s, c_s, phi_s = UK_elastic_string(x, params)
    
    F_idx   = 1  
    ts      = 0.
    te      = 1e-2
    Fs      = 0.
    Fe      = 5.
    params  = (t, ts, te, Fs, Fe)
    
    Fext_s = UK_apply_force(Nt, Nx, phi_s, F_idx, UK_ramp_force, params)
    
    # BOARD MODEL -------------------------------------------------------
    
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
    
    Fext_b  = UK_apply_force(Nt, 1, phi_b)
    
    # OVERALL MODEL -----------------------------------------------------
    
    m_tuple     = (m_s, m_b) 
    k_tuple     = (k_s, k_b)
    c_tuple     = (c_s, c_b)
    Fext_tuple  = (Fext_s, Fext_b)
    phi_tuple   = (phi_s, phi_b)
    
    M, K, C, phi, Fext = UK_give_overall_model(m_tuple, k_tuple, c_tuple, 
                                         Fext_tuple, phi_tuple)
    
    # CONSTRAINTS -------------------------------------------------------
    
    if coupled:
        constraints = (UK_constraint_fixed(0, 0), 
                       UK_constraint_contact((0,1), (2,0)))
    else:
        constraints = (UK_constraint_fixed(0, 0), 
                       UK_constraint_fixed(0, 2))
    
    A, b, phi_c = UK_give_A_b(phi_tuple, constraints)
        
    # W, V, Wc, Vc MATRICES ---------------------------------------------
    
    W, V, Wc, Vc = UK_give_W_V(A, M, b)
    
    # INITIAL CONDITIONS ------------------------------------------------
    
    initial = (UK_initial_rest(0), UK_initial_rest(1))
    
    q0, qd0, qdd0, Fc0 = UK_give_initial_state(phi_tuple, initial)
    
    # RETURN ------------------------------------------------------------
    
    return M, K, C, A, b, W, V, Wc, Vc, phi, phi_c, Fext, q0, qd0 ,qdd0, Fc0, h



# def toy_3el_model():
#     print('Model used : toy_model')
    
#     # TIME ARRAY -------------------------------------------------------
    
#     h       = 1e-5
#     t       = np.arange(0,10, h) 
    
#     Nt = t.size
    
#     # STRING MODEL -----------------------------------------------------
    
#     L_s = 0.65
    
#     x_s   = np.array([L_s])  
#     Nx_s  = x_s.size
    
#     params_s          = {}
#     params_s['Nn_s']  = 200
#     params_s['T']     = 73.9
#     params_s['L']     = L_s 
#     params_s['rho_l'] = 3.61e-3
#     params_s['B']     = 0.
#     params_s['etaF']  = 0.
#     params_s['etaA']  = 0.
#     params_s['etaB']  = 0.
    
#     m_s, k_s, c_s, phi_s = UK_elastic_string(x_s, params_s)
    
#     F_idx   = 1  
#     ts      = 0.
#     te      = 1e-2
#     Fs      = 0.
#     Fe      = 5.
#     params_s  = (t, ts, te, Fs, Fe)
    
#     Fext_s = UK_apply_force(Nt, Nx_s, phi_s, F_idx, UK_ramp_force, params_s)
    
#     # BRIDGE MODEL -----------------------------------------------------
    
#     L_b     = 6e-2
#     Nx_b    = 10
#     x_b     = np.linspace(0, L_b, Nx_b)
    
#     w_b = 6e-2
#     h_b = 0.5e-2
#     I_b = utils.give_I_rectangle(w_b, h_b)
    
#     params_b          = {}
#     params_b['Nn_b']  = 4 
#     params_b['L']     = L_b 
#     params_b['E']     = 3e9
#     params_b['I']     = I_b
#     params_b['S']     = w_b*h_b
#     params_b['rho']   = 800
    
#     m_b, k_b, c_b, phi_b = UK_beam(x_b, params_b)
    
#     Fext_b = UK_apply_force(Nt, Nx_b, phi_b)
    
#     # BOARD MODEL -------------------------------------------------------
    
#     h_p     = 2e-3
#     a_p     = 0.5 
#     b_p     = 0.3
#     E_p     = 0.5e9
#     I_x_p   = utils.give_I_rectangle(b_p, h_p) 
#     I_y_p   = utils.give_I_rectangle(a_p, h_p)
    
#     x_p_bridge  = 0.25*a_p
#     x_p         = np.array([0, x_p_bridge, a_p])
#     y_p         = x_b + b_p/2 - L_b/2
#     y_p         = np.concatenate(([0],y_p, [b_p]))
    
#     x_p, y_p    = utils.make_grid(x_p, y_p)
#     Nx_p        = x_p.size 
    
#     params_p = {
#         'Nm_p'  : 12, 
#         'Nn_p'  : 12,
#         'h'     : h_p, 
#         'a'     : a_p,
#         'b'     : b_p,
#         'rho'   : 600,
#         'D'     : (E_p*I_x_p, E_p*I_y_p, E_p*I_x_p/2, E_p*I_y_p/2),
#         }
    
#     m_p, k_p, c_p, phi_p = UK_board(x_p, y_p, params_p)
    
#     Fext_p = UK_apply_force(Nt, Nx_p, phi_p)
    
#     # OVERALL MODEL -----------------------------------------------------
    
#     m_tuple     = (m_s, m_b, m_p) 
#     k_tuple     = (k_s, k_b, k_p)
#     c_tuple     = (c_s, c_b, c_p)
#     Fext_tuple  = (Fext_s, Fext_b, Fext_p)
#     phi_tuple   = (phi_s, phi_b, phi_p)
    
#     M, K, C, phi, Fext = UK_give_overall_model(m_tuple, k_tuple, c_tuple, 
#                                          Fext_tuple, phi_tuple)
    
#     # CONSTRAINTS -------------------------------------------------------
    
#     idx_b_middle    = Nx_b//2 + Nx_b%2
#     idx_p_bridge    = 1 
    
#     constraints = (UK_constraint_fixed(0, 0), 
#                    UK_constraint_contact((0, 1), (2, idx_b_middle)),)
    
#     for i in range(x_b.size):
#         constraints += (UK_constraint_contact((1, 2), (i, idx_p_bridge + i)),)
    
#     A, b, phi_c = UK_give_A_b(phi_tuple, constraints)
    
#     # W, V, Wc, Vc MATRICES ---------------------------------------------
    
#     W, V, Wc, Vc = UK_give_W_V(A, M, b)
    
#     # INITIAL CONDITIONS ------------------------------------------------
    
#     initial = (UK_initial_rest(0), UK_initial_rest(1), UK_initial_rest(2))
    
#     q0, qd0, qdd0, Fc0 = UK_give_initial_state(phi_tuple, initial)
    
#     # RETURN ------------------------------------------------------------
    
#     return M, K, C, A, b, W, V, Wc, Vc, phi, phi_c, Fext, q0, qd0 ,qdd0, Fc0, h



def toy_2el_model():
    print('Model used : toy_model')
    
    # TIME ARRAY -------------------------------------------------------
    
    h       = 1e-5
    t       = np.arange(0,10, h) 
    
    Nt = t.size
    
    # STRING MODEL -----------------------------------------------------
    
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
    
    m_s, k_s, c_s, phi_s = UK_elastic_string(x_s, params_s)
    
    F_idx   = 0  
    ts      = 0.
    te      = 1e-2
    Fs      = 0.
    Fe      = 5.
    params_s  = (t, ts, te, Fs, Fe)
    
    Fext_s = UK_apply_force(Nt, Nx_s, phi_s, F_idx, UK_ramp_force, params_s)
    
    # BOARD MODEL -------------------------------------------------------
    
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
    
    m_p, k_p, c_p, phi_p = UK_board(x_p, y_p, params_p)
    
    Fext_p = UK_apply_force(Nt, Nx_p, phi_p)
    
    # OVERALL MODEL -----------------------------------------------------
    
    m_tuple     = (m_s, m_p) 
    k_tuple     = (k_s, k_p)
    c_tuple     = (c_s, c_p)
    Fext_tuple  = (Fext_s, Fext_p)
    phi_tuple   = (phi_s, phi_p)
    
    M, K, C, phi, Fext = UK_give_overall_model(m_tuple, k_tuple, c_tuple, 
                                         Fext_tuple, phi_tuple)
    
    # CONSTRAINTS -------------------------------------------------------
    
    constraints = (UK_constraint_contact((0, 1), (1, 1)),)
    
    A, b, phi_c = UK_give_A_b(phi_tuple, constraints)
    
    # W, V, Wc, Vc MATRICES ---------------------------------------------
    
    W, V, Wc, Vc = UK_give_W_V(A, M, b)
    
    # INITIAL CONDITIONS ------------------------------------------------
    
    initial = (UK_initial_rest(0), UK_initial_rest(1))
    
    q0, qd0, qdd0, Fc0 = UK_give_initial_state(phi_tuple, initial)
    
    # RETURN ------------------------------------------------------------
    
    return M, K, C, A, b, W, V, Wc, Vc, phi, phi_c, Fext, q0, qd0 ,qdd0, Fc0, h
    


'''
-------------------------------------------------------------
                   MAIN FUNCTION
-------------------------------------------------------------
'''

def main() -> None:
    
    import disp
    
    save = True

    # Getting system's model and initial variables 
    M, K, C, A, b, W, V, Wc, Vc, phi, phi_c, Fext, q, qd ,qdd, Fc, h = \
        toy_2el_model()
        # ANTUNES_2017(True)
    
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
    
    # Save results
    
    if save:
        data.save_simulation(M, K, C, A, b, W, V, Wc, Vc, phi, phi_c, Fext, x,\
                             xd ,xdd, Fc, h, save_dir='../results')
    
    # Results display
    disp.set_gui_qt()
    disp.summary_plot(x, Fc_phys, h)
    
'''
-------------------------------------------------------------
-------------------------------------------------------------
'''

if __name__ == "__main__":
     main()