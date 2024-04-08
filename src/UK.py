# -*- coding: utf-8 -*-

'''
-------------------------------------------------------------
                   UK-SIMULATION FUNCTIONS
-------------------------------------------------------------
'''

# LIBRARIES -------------------------------------------------------------------

import numpy as np
import scipy as sp
import src.utils as utils

# FUNCTIONS -------------------------------------------------------------------


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
Outputs : q           : (Nn)          current modal response 
          qd          : (Nn)          current modal response's velocity
          qdd         : (Nn)          current modal response's acceleration
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
Description :   makes the violation elimination technique (Yoon et. al, 1994)
                at a given time step.
 
Inputs :  q             : (Nn)     current modal response at a given time 
                                   step
          qd            : (Nn)     current modal response's velocity at a 
                                   given time step
          constraints   : (Nm)     tuple of dictionnaries containing the 
                                   constraint parameters
          phi_tuple     : (Ns)     tuple containing all the modeshapes (Nx_i, 
                                                                        Nn_i)
          A             : (Nm, Nn) modal constraint matrix
        
Outputs : q             : (Nn)     corrected modal response 
          qd            : (Nn)     corrected modal response's velocity
'''
def violation_elim(q, qd, constraints, phi_c, A):
    Nm = A.shape[0]
    
    Amp  =  give_MP_right_inv(A) 
    
    viol_q = np.zeros(Nm)
    i = 0
    while i < len(constraints):
        const = constraints[i]
        if const['type'] == 'fixed':
            viol_q[i] = phi_c[i] @ q
            i += 1
        elif const['type'] == 'contact':
            viol_q[i] += phi_c[i] @ q
            i += 1
        elif const['type'] == 'surface_contact':
            for j in range(len(const['idx_1'])):
                viol_q[i+j] += phi_c[i+j] @ q 
                i += 1
                
    q   -= Amp @ viol_q 
    
    viol_qd = np.zeros(Nm) 
    
    qd -= Amp @ viol_qd
    
    return q, qd
        

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
    Fc_phys = Fc @ give_MP_right_inv(phi_c)
    
    return x, xd, xdd, Fc_phys


'''
Description :   gives the Moore-Penrose right pseudoinverse of a given matrix
                M.

Inputs  : M                 : 2d numpy array
Outputs : M^T(MM^T)^(-1)    : 2d numpy array  
'''
def give_MP_right_inv(M):
    if np.any(M != 0):
        Mpi = M.T @ np.linalg.inv(M @ M.T)
    else:
        Mpi = M.T
    return Mpi

'''
Description :   gives the Moore-Penrose left pseudoinverse of a given matrix
                M.

Inputs  : M                 : 2d numpy array
Outputs : (M^TM)^(-1)M^T    : 2d numpy array  
'''
def give_MP_left_inv(M):
    if np.any(M != 0):
        Mpi = np.linalg.inv(M.T @ M) @ M.T 
    else:
        Mpi = M.T
    return Mpi


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
        V           = np.zeros((Na, Nn))
        Wc          = np.zeros((Na, Nn, Nn))
        Vc          = np.zeros((Na, Nn))
        for i in range(Na):
            if np.any(A[i] != 0):
                B       = A[i] @ M_sq_inv
                Bp      = give_MP_right_inv(B)
                MB      = M_sq_inv @ Bp
                W[i]    = I - MB @ A[i]
                V[i]    = MB @ b[i]
                Wc[i]   = -M_sq @ Bp @ A[i]
                Vc[i]   = M_sq @ Bp @ b[i]
            else:
                W[i] = I
    # Case where Bp is computed on-line
    else:
        if np.any(A != 0):
            M_sq        = M.power(1/2)
            M_sq_inv    = M.power(-1/2)
            B           = A @ M_sq_inv 
            Bp          = B.T @ np.linalg.inv((B @ B.T))
            MB          = M_sq_inv @ Bp
            W           = np.eye(M.shape[0]) - MB @ A
            V           = MB @ b
            Wc          = -M_sq @ Bp @ A
            Vc          = M_sq @ Bp @ b
        else:
            Nm,Nn = A.shape
            W   = np.eye(Nn)
            V   = np.zeros((Nn))
            Wc  = np.zeros((Nn, Nn))
            Vc  = np.zeros((Nn))
        
    return W, V, Wc, Vc


'''
Description :   gives UK's model modal physical force matrix Fext for 
                a given physical force matrix Fext_phys and modal deformation 
                matrix.

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
    info_f = {
        "info_type"     : "force",
        "force_type"    : "ramp",
        "params"        : (ts, te, Fs, Fe)}
    return F_ramp, info_f
    

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
    Fext_phys = sp.sparse.csr_array((Nt,Nx))
    if F_fun:
        Fext_phys[:, F_idx], info_f  = F_fun(*params)
        info_f['F_idx'] = F_idx
    else:
        info_f = {
            "info_type" : "force",
            "force_type" : "zero"}
    Fext = give_Fext(Fext_phys, phi)
    return Fext, Fext_phys, info_f


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
    
    info_s = {
        "info_type"   : "subsystem",
        "subsystem"   : "elastic_string",
        "params"      : params,
        "x"           : x
        }
    
    return m_s, k_s, c_s, phi_s, info_s


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
    Nn_b    = params['Nn_b'] - 1 #-1 to take because of the 0 rigid body mode 
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
    m_b_0       = L*rho*S
    m_b         = np.append([m_b_0], m_b)
    
    zeta_b = np.zeros(1+ 2*Nn_b)
    
    phi_b_1         = np.cos(np.outer(x,k_b)) + np.cosh(np.outer(x,k_b))
    phi_b_2         = np.sin(np.outer(x,k_b)) + np.sinh(np.outer(x,k_b))
    phi_b           = np.zeros((Nx,2*Nn_b))
    phi_b[:,::2]    = phi_b_1 
    phi_b[:,1::2]   = phi_b_2 
    phi_b_0         = np.ones((Nx,1))
    phi_b           = np.hstack((phi_b_0, phi_b)) 
    
    w_b     = np.repeat(w_b,2) 
    w_b_0   = 0
    w_b     = np.append([w_b_0], w_b)
    
    k_b     = m_b*(w_b)**2
    
    c_b     = 2*m_b*w_b*zeta_b
    
    info_b = {
        "info_type"   : "subsystem",
        "subsystem"   : "beam",
        "params"      : params,
        "x"           : x
        }
    
    return m_b, k_b, c_b, phi_b, info_b


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
    
    info_p = {
        "info_type"   : "subsystem",
        "subsystem"   : "board",
        "params"      : params,
        "x"           : x,
        "y"           : y
        }
    
    return m_p, k_p, c_p, phi_p, info_p


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
    f_p     = params['f_p']
    zeta_p  = params['zeta_p']
    m_p     = params['m_p']
    
    Nn_p    = f_p.size 
    w_p     = 2*np.pi*f_p
    k_p     = m_p*(w_p)**2
    c_p     = 2*m_p*w_p*zeta_p
    phi_p   = np.ones((1,Nn_p))  # body modes normalized at the bridge location
    
    info_p = {
        "info_type"   : "subsystem",
        "subsystem"   : "board_modal",
        "params"      : params
        }
    
    return m_p, k_p, c_p, phi_p, info_p


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

Inputs :    idx_sys : tuple with the indices of the two constrained systems
            idx_x   : tuple with indices of the two positions 
                        
Outputs :   constraint : corresponding constraint dictionnary 
'''    
def UK_constraint_contact(idx_sys, idx_x):
    constraint = {'type' : 'contact',
                  'idx_sys' : idx_sys,
                  'idx_x' : idx_x} 
    return constraint



def UK_constraint_surface_contact(idx_sys, idx_1, idx_2):
    constraint = {'type' : 'surface_contact',
                  'idx_sys' : idx_sys,
                  'idx_1' : idx_1,
                  'idx_2' : idx_2}
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
    info_c = []
    Nn_tuple = (0,)
    for phi in phi_tuple:
        Nn_tuple += (Nn_tuple[-1] + phi.shape[-1],) 
    Nn          = Nn_tuple[-1]
    Nm          = len(constraints)
    
    if Nm==0:
        A       = np.zeros((1,Nn))
        b       = np.zeros((1))
        phi_c   = np.zeros((1,Nn)) 
    else:
        for i,c in enumerate(constraints):
            if c['type'] == 'contact':
                a_temp      = np.zeros((1,Nn))  
                b_temp      = np.zeros(1)
                phi_c_temp  = np.zeros((1,Nn))
                idx_sys_1, idx_sys_2 = c['idx_sys']
                idx_x_1, idx_x_2     = c['idx_x']
                a_temp[0,Nn_tuple[idx_sys_1]:Nn_tuple[idx_sys_1+1]]  = \
                    phi_tuple[idx_sys_1][idx_x_1]
                a_temp[0,Nn_tuple[idx_sys_2]:Nn_tuple[idx_sys_2+1]]  = \
                    -phi_tuple[idx_sys_2][idx_x_2]
                b_temp      = np.zeros(1)
                phi_c_temp[0,Nn_tuple[idx_sys_1]:Nn_tuple[idx_sys_1+1]] = \
                    phi_tuple[idx_sys_1][idx_x_1]
                info_c += [{"info_type"     : "constraint",
                            "constraint"    : c}]
            elif c['type'] == 'fixed':
                a_temp      = np.zeros((1,Nn))  
                b_temp      = np.zeros(1)
                phi_c_temp  = np.zeros((1,Nn))
                idx_sys                                     = c['idx_sys']
                idx_x                                       = c['idx_x']
                a_temp[0,Nn_tuple[idx_sys]:Nn_tuple[idx_sys+1]]  = \
                    phi_tuple[idx_sys][idx_x]
                b_temp      = np.zeros(1)
                phi_c_temp[0,Nn_tuple[idx_sys]:Nn_tuple[idx_sys+1]] = \
                    phi_tuple[idx_sys][idx_x]
                info_c += [{"info_type"     : "constraint",
                            "constraint"    : c}]
            elif c['type'] == 'surface_contact':
                a_temp      = np.zeros((len(c['idx_1']), Nn))  
                b_temp      = np.zeros((len(c['idx_1'])))
                phi_c_temp  = np.zeros((len(c['idx_1']), Nn))
                idx_sys_1, idx_sys_2    = c['idx_sys']
                idx_1                   = c['idx_1']
                idx_2                   = c['idx_2']
                for j in range(len(idx_1)):
                    a_temp[j,Nn_tuple[idx_sys_1]:Nn_tuple[idx_sys_1+1]]     = \
                        phi_tuple[idx_sys_1][idx_1[j]]
                    a_temp[j,Nn_tuple[idx_sys_2]:Nn_tuple[idx_sys_2+1]]     = \
                        -phi_tuple[idx_sys_2][idx_2[j]]
                    phi_c_temp[j,Nn_tuple[idx_sys_1]:Nn_tuple[idx_sys_1+1]] = \
                        phi_tuple[idx_sys_1][idx_1[j]]
                info_c += [{"info_type"     : "constraint",
                            "constraint"    : c}]
            if 'A' not in locals():
                A       = a_temp
                b       = b_temp
                phi_c   = phi_c_temp
            else:
                A       = np.vstack((A, a_temp))
                b       = np.concatenate((b, b_temp))  
                phi_c   = np.vstack((phi_c, phi_c_temp))
    return A, b, phi_c, info_c


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
    info_i = []
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
            info_i += [{
                "info_type" : "initial",
                "initial" : init}]
    
    return q0, qd0, qdd0, Fc0, info_i

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
def UK_give_overall_model(m_tuple, k_tuple, c_tuple, Fext_tuple, 
                          Fext_phys_tuple, phi_tuple):
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
    
    Fext_phys = sp.sparse.hstack(Fext_phys_tuple)
    Fext_phys = sp.sparse.csr_array(Fext_phys)
    
    PHI             = np.zeros((Nx, Nn))
    for (i,phi) in enumerate(phi_tuple):
        PHI[Nx_tuple[i]:Nx_tuple[i+1], Nn_tuple[i]:Nn_tuple[i+1]] = phi
    
    return M, K, C, PHI, Fext, Fext_phys
