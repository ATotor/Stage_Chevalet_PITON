# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp

import os
from os.path import isdir 
    

def save_simulation(M, K, C, A, b, W, V, Wc, Vc, phi, phi_c, Fext, Fext_phys, 
                    x, xd, xdd, Fc, Fc_phys, h, save_dir='./results'):
    if not isdir(save_dir): 
        os.mkdir(save_dir)
    
    n = 1
    save_dir_temp = save_dir + '/simulation_' + str(n) 
    
    while(isdir(save_dir_temp)):
        n += 1
        save_dir_temp = save_dir + '/simulation_' + str(n)
    save_dir = save_dir_temp
    os.mkdir(save_dir)
    
    save_name_np = save_dir+'/np_arr'
    np.savez_compressed(save_name_np, 
                        A       = A,
                        b       = b,
                        W       = W,
                        V       = V,
                        Wc      = Wc,
                        Vc      = Vc,
                        phi     = phi,
                        phi_c   = phi_c,
                        x       = x,
                        xd      = xd,
                        xdd     = xdd,
                        Fc_phys = Fc_phys,
                        Fc      = Fc,
                        h       = h)
    save_sparse(M, save_dir, 'M')
    save_sparse(K, save_dir, 'K')
    save_sparse(C, save_dir, 'C')
    save_sparse(Fext, save_dir, 'Fext')
    save_sparse(Fext_phys, save_dir, 'Fext_phys')
    
    print("Saved simulation data in : " + save_dir)

    

def save_sparse(sparse, save_dir='./results', save_name="sparse"):
    if not isdir(save_dir): 
        os.mkdir(save_dir)
    
    files = [f for f in os.listdir(save_dir)]
    
    n = 1
    valid = False
    while(not valid):
        valid = True
        for f in files:
            if (f[-4:] == ".npz") and (f[:len(save_name)]==save_name):
                idx_ = -6
                while(f[idx_] != "_"):
                    idx_ -= 1
                temp = int(f[idx_+1:-4])
                if n == temp:
                    n += 1
                    valid = False
    
    save_name = save_dir+'/'+save_name+'_'+str(n)
    sp.sparse.save_npz(save_name, sparse)
    print("Saved sparse array in : " + save_name)
    
    
def save_array(arr, save_dir='./results', save_name='arr'):
    if not isdir(save_dir): 
        os.mkdir(save_dir)
    
    files = [f for f in os.listdir(save_dir)]
    
    n = 1
    valid = False
    while(not valid):
        valid = True
        for f in files:
            if (f[-4:] == ".npz") and (f[:len(save_name)]==save_name):
                idx_ = -6
                while(f[idx_] != "_"):
                    idx_ -= 1
                temp = int(f[idx_+1:-4])
                if n == temp:
                    n += 1
                    valid = False
    
    save_name  = save_dir+'/'+save_name+'_'+str(n)
    np.save(save_name, arr)
    print("Saved array in : " + save_name)
    
        
def load_simulation(n, load_dir='./results'):
    load_dir = load_dir+'/simulation_'+str(n)
    print('Loading simulation data from folder : ' + load_dir)
    loaded      = np.load(load_dir + "/np_arr.npz", allow_pickle=True)
    M           = load_sparse(load_dir = load_dir, load_name='M') 
    K           = load_sparse(load_dir = load_dir, load_name='K')
    C           = load_sparse(load_dir = load_dir, load_name='C')
    Fext        = load_sparse(load_dir = load_dir, load_name='Fext')
    Fext_phys   = load_sparse(load_dir = load_dir, load_name='Fext_phys')
    
    return M, K, C, loaded['A'], loaded['b'], loaded['W'],loaded['V'],\
        loaded['Wc'], loaded['Vc'], loaded['phi'], loaded['phi_c'], Fext,\
            Fext_phys, loaded['x'], loaded['xd'], loaded['xdd'], loaded['Fc'],\
                loaded['Fc_phys'], loaded['h'] 
    
        
def load_sparse(n=1, load_dir='./results', load_name='sparse'):
    name = load_dir+'/'+load_name+'_'+str(n)+'.npz'
    print('Loading file : "'+name+'"')
    sparse = sp.sparse.load_npz(name)
    return sparse


def load_array(n=1, load_dir='./results', load_name='arr'):
    name = load_dir+'/'+load_name+'_'+str(n)+'.npy'
    print('Loading file : "'+name+'"')
    arr = np.load(name)
    
    return arr
        



