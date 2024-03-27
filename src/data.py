# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp

import os
from os.path import isdir 
    

def save_simulation(info, M, K, C, A, b, W, V, Wc, Vc, phi, phi_c, Fext, 
                    Fext_phys, x, xd, xdd, Fc, Fc_phys, h, 
                    save_dir='./results'):
    if not isdir(save_dir): 
        os.mkdir(save_dir)
    
    n = 1
    save_dir_temp = save_dir + '/simulation_' + str(n) 
    
    while(isdir(save_dir_temp)):
        n += 1
        save_dir_temp = save_dir + '/simulation_' + str(n)
        
    info = info + ({
        "info_type" : "file",
        "n" : n},)
    
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
    
    write_info_file(info, save_dir)
    
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
        

def write_elastic_string(info):
    return 'Elastc string\n\t\tAn elastic string description ...\n'


def write_beam(info):
    return 'Beam\n\t\tA beam description ...\n'


def write_board(info):
    return 'Board\n\t\tA board description ...\n'


def write_board_modal(info):
    return 'Modal board\n\t\tA modal board description ...\n'


def write_ramp(info):
    return 'Ramp\n\t\tA ramp force description...\n'


def write_fixed(info):
    return 'Fixed\n\t\tA fixed constraint description ...\n'


def write_contact(info):
    return 'Contact\n\t\tA contact constraint description ...\n'


def write_rest(info):
    return 'Rest\n\t\tA rest initial condition description ...\n'


def write_info_file(info, write_dir):
    n = info[-1]["n"] 
    text = "----------- simulation_"+str(n)+" -----------\n\nDESCRIPTION\n"+\
        "\nSUBSYSTEMS\n"
    n = 1
    for inf in info[:-1]:
        if inf["info_type"]=="subsystem":
            text = text + '\tS' + str(n)+"-"
            if inf["subsystem"]=="elastic_string":
                text = text + write_elastic_string(inf)
            elif inf["subsystem"]=="beam":
                text = text + write_beam(inf)
            elif inf["subsystem"]=="board":
                text = text + write_board(inf)
            elif inf["subsystem"]=="board_modal":
                text = text + write_board(inf)
            n += 1
    
    text = text + "\nEXTERNAL FORCES\n"
    n = 1
    for inf in info[:-1]:
        if inf["info_type"]=="force":
            text = text + '\tF' + str(n)+"-"
            if inf["force_type"]=="ramp":
                text = text + write_ramp(inf)
            n += 1
            
    text = text + "\nCONSTRAINTS\n"
    n = 1
    for inf in info[:-1]:
        if inf["info_type"]=="constraint":
            text = text + '\tC' + str(n)+"-"
            if inf["constraint"]["type"]=="fixed":
                text = text + write_fixed(inf)
            elif inf["constraint"]["type"]=="contact":
                text = text + write_contact(inf)
            n += 1
            
    text = text + "\nINITIAL CONDITIONS\n"
    n = 1
    for inf in info[:-1]:
        if inf["info_type"]=="initial":
            text = text + '\tI' + str(n)+"-"
            if inf["initial"]["type"]=="rest":
                text = text + write_rest(inf)
            n += 1
    
    with open(write_dir+'/info.txt', 'w', encoding='utf-8') as f:
        f.write(text)


