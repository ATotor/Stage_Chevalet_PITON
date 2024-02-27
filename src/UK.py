# -*- coding: utf-8 -*-

import numpy as np


def UK_step(M, K, C, A, b, Fext, q, qd, qdd):
    
    return


# Inputs  : Fext_phys   : (Nt,Nx) external force matrix, 
#           phi         : (Nn,Nx) modal deformation vector.
# Outputs : Fext        : (Nt,Nn) external modal force matrix. 
def give_Fext(Fext_phys, phi, dx):
    Nt,Nx   = Fext_phys.shape
    Nn,_    = phi.shape
    Fext    = np.zeros((Nt, Nn))
    
    for n,phi_n in enumerate(phi):
        print('hi')
        print(Fext_phys*phi_n)
        print(n)
        Fext[:,n] = np.trapz(Fext_phys*phi_n, dx=dx)
    
    return Fext


def ANTUNES_2017():
    h       = 1e-5
    t       = np.arange(0,10, h)
    dx      = 1e-3 
    L       = 0.65
    x       = np.arange(0,L, dx)
    n       = np.arange(200)
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
    tF      = 1e-2
    
    Nn = n.size
    Nx = x.size
    Nt = t.size
    
    
    p = (2*n-1)*np.pi/(2*L)
    f = ct/(2*np.pi)*p*(1+B/(2*T)*p**2)    
    w = 2*np.pi*f
    
    phi = np.zeros((Nn, Nx))
    for i in n:
        phi[i] = np.sin(p[i]*x)
    
    Fext_phys               =  np.zeros((Nt, Nx))
    Fext_phys[t<tF,idxF]   =  5*t[t<tF]/t[t<tF][-1]
    Fext                    = give_Fext(Fext_phys, phi, dx)
    
    m = rho_l*np.trapz(phi**2, dx=dx)
    
    zeta = 1/2*(T*(etaF+etaA/w)+etaB*B*p**2)/(T+B*p**2) 
    
    return Fext, m, zeta, w
    #return M, K, C, A, b, Fext, q, qd ,qdd, h, dx
    

def main() -> None:
    Fext, m, zeta, f = ANTUNES_2017()
    
    import matplotlib.pyplot as plt
    
    plt.figure()
    plt.plot(f)
    

if __name__ == "__main__":
    main()
    