# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp


def UK_step(M, K, C, A, b, Fext, q, qd, qdd):
    
    return


# Inputs  : Fext_phys   : (Nx,) external force matrix, 
#           phi         : (Nn,Nx) modal deformation vector.
# Outputs : Fext        : (Nn,) external modal force matrix. 
def give_Fext(Fext_phys, phi, dx):
    return np.trapz(phi*Fext_phys, dx=dx)


def ANTUNES_2017():
    h       = 1e-5
    t       = np.arange(0,10, h) 
    tF      = 1e-2
    
    Nt = t.size
    
    # String model
    
    L       = 0.65
    dx      = 1e-3 
    x       = np.arange(0,L, dx)
    n       = np.arange(200)+1
    T       = 73.9
    rho_l   = 3.61e-3
    ct      = np.sqrt(T/rho_l)
    B       = 4e-5
    etaF    = 7e-5
    etaA    = 0.9
    etaB    = 2.5e-2
    xE      = 0.9*L
    xF      = 0.33*L
    idxF    = int(np.round(xF/dx))
    
    Nn = n.size
    Nx = x.size
    
    p = (2*n-1)*np.pi/(2*L)
    f_s = ct/(2*np.pi)*p*(1+B/(2*T)*p**2)    
    w_s = 2*np.pi*f_s
    
    phi_s = np.zeros((Nn, Nx))
    for i in range(n.size):
        phi_s[i] = np.sin(p[i]*x)
    
    Fext_phys_s               =  np.zeros((Nt, Nx))
    Fext_phys_s[t<tF,idxF]   =  5*t[t<tF]/t[t<tF][-1]
    Fext_s = np.zeros((Nt, Nn))
    for i in range(Nn):
        Fext_s[i] = give_Fext(Fext_phys_s[i], phi_s, dx)
    
    m_s = rho_l*np.trapz(phi_s**2, dx=dx)
    
    zeta_s = 1/2*(T*(etaF+etaA/w_s)+etaB*B*p**2)/(T+B*p**2) 
    
    k_s = m_s*(w_s)**2
    
    c_s = 2*m_s*w_s*zeta_s
    
    
    # Board model
    
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
    
    Fext_b  = np.zeros(f_b.size)
    
    # Overall model
    
    m = np.concatenate((m_s, m_b)) 
    k = np.concatenate((k_s, k_b))
    c = np.concatenate((c_s, c_b))
    
    M = sp.sparse.dia_array(np.diag(m))
    K = sp.sparse.dia_array(np.diag(k))
    C = sp.sparse.dia_array(np.diag(c))
    
    Fext = np.concatenate((Fext_s,Fext_b))
    
    # Constraints
    
    A = 
    
    b = 
    
    # Initial conditions
    
    q0      = np.zeros(M.shape[0])
    qd0     = np.zeros(M.shape[0])
    qdd0    = np.zeros(M.shape[0]) 
    
    return M, K, C, A, b, Fext, q0, qd0 ,qdd0, h, dx
    

def main() -> None:
    import matplotlib.pyplot as plt
    
    Fext, m, zeta, f, h, dx, Fext_phys = ANTUNES_2017()
    
    plt.figure()
    plt.plot(f)
    plt.figure()
    plt.plot(np.arange(0,Fext.shape[0]*h,h)[:500], Fext_phys[:500])
    

if __name__ == "__main__":
    main()
    