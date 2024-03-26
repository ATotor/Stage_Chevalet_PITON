# -*- coding: utf-8 -*-

import numpy as np
import os
from os.path import isdir 

def save_simulation(M, K, C, A, b, W, V, Wc, Vc, phi, phi_c, Fext, x, xd ,xdd, 
                    Fc, h, save_dir='./results'):
    if not isdir(save_dir): 
        os.mkdir(save_dir)
    
    files = [f for f in os.listdir(save_dir)]
    
    n = 1
    valid = False
    while(not valid):
        valid = True
        for f in files:
            if f[-4:] == ".npz":
                idx_ = -6
                while(f[idx_] != "_"):
                    idx_ -= 1
                temp = int(f[idx_+1:-4])
                if n == temp:
                    n += 1
                    valid = False
    
    np.savez_compressed(save_dir+'/simulation_'+str(n), 
                        M       = M, 
                        K       = K,
                        C       = C,
                        A       = A,
                        b       = b,
                        W       = W,
                        V       = V,
                        Wc      = Wc,
                        Vc      = Vc,
                        phi     = phi,
                        phi_c   = phi_c,
                        Fext    = Fext,
                        x       = x,
                        xd      = xd,
                        xdd     = xdd,
                        Fc      = Fc,
                        h       = h)
    
    
def load_simulation(n, load_dir='./results'):
    name = load_dir+'/simulation_'+str(n)+'.npz'
    print('Loading file : "'+name+'"')
    loaded = np.load(name, allow_pickle=True)
    
    return loaded['M'], loaded['K'], loaded['C'], loaded['A'], loaded['b'],\
        loaded['W'],loaded['V'], loaded['Wc'], loaded['Vc'], loaded['phi'],\
        loaded['phi_c'], loaded['Fext'], loaded['x'], loaded['xd'],\
        loaded['xdd'], loaded['Fc'], loaded['h'] 

