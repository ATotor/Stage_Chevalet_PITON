# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp

import os
from os.path import isdir 
from scipy.io.wavfile import write
    

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
    
    write_info_file(n, info, save_dir)
    
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

def pop_info(info):
    info.pop('info_type')
    
def write_elastic_string(info):
    text = '''Elastic string
\t\tElastic string vibrating in 1 transverse direction.
'''
    for a,b in info.items():
        text += f'\t\t{a} : {b}\n'
    return text


def write_beam(info):
    text = '''Beam
\t\tElastic beam vibrating in 1 transverse direction.
'''
    for a,b in info.items():
        text += f'\t\t{a} : {b}\n'
    return text


def write_board(info):
    text = '''Board
\t\tElastic orthotropic board vibrating in its transverse direction.
'''
    for a,b in info.items():
        text += f'\t\t{a} : {b}\n'
    return text


def write_board_modal(info):
    text = '''Modal board
\t\tElastic modal board and bridge obtained by experimental measurements.
'''
    for a,b in info.items():
        text += f'\t\t{a} : {b}\n'
    return text


def write_ramp(info):
    text = 'Ramp\n\t\tA ramp force starting at t=params[0] (amplitude '+\
    'F=params[2]), ending at t=params[1] (amplitude F=params[3]).\n'
    for a,b in info.items():
        text += f'\t\t{a} : {b}\n'
    return text

def write_gaussian(info):
    text = 'Gaussian\n\t\tA gaussian force with peak at t=params[0]'+\
        ' and width w=params[1] (peak F=params[2]).\n'
    for a,b in info.items():
        text += f'\t\t{a} : {b}\n'
    return text

def write_zero(info):
    text = 'Zero\n\t\tNo force is applied\n'
    for a,b in info.items():
        text += f'\t\t{a} : {b}\n'
    return text


def write_fixed(info):
    text = 'Fixed\n\t\tA fixed point cannot accelerate : xdd=0.\n'
    for a,b in info.items():
        text += f'\t\t{a} : {b}\n'
    return text 


def write_contact(info):
    text = 'Contact\n\t\tA contact constraint means that the two points from'+\
    'the two subsystems have the same acceleration : xdd1=xdd2.\n'
    for a,b in info.items():
        text += f'\t\t{a} : {b}\n'
    return text

def write_surface_contact(info):
    text = 'Surface contact\n\t\tA surface contact constraint means that '+\
        'two points from two subsystems have the same acceleration : '+\
            'xdd1=xdd2.\n'
    for a,b in info.items():
        text += f'\t\t{a} : {b}\n'
    return text


def write_rest(info):
    text = 'Rest\n\t\tA rest initial condition at a point means that '+\
    'x=xd=xdd=0 at t=0.\n'
    for a,b in info.items():
        text += f'\t\t{a} : {b}\n'
    return text


def write_info_file(n, info, write_dir):
    text = f"""\
----------- simulation_{n} -----------
        
DESCRIPTION
    This file containts additionnal information about the simulation.\
 All units are in S.I.

SUBSYSTEMS
\
    """
    n = 0
    for inf in info.copy():
        if inf["info_type"]=="subsystem":
            pop_info(inf)
            text = text + '\tS' + str(n)+"-"
            subsystem = inf.pop("subsystem")
            if subsystem=="elastic_string":
                text = text + write_elastic_string(inf)
            elif subsystem=="beam":
                text = text + write_beam(inf)
            elif subsystem=="board":
                text = text + write_board(inf)
            elif subsystem=="board_modal":
                text = text + write_board_modal(inf)
            info.remove(inf)
            n += 1
    
    text = text + "\nEXTERNAL FORCES\n"
    n = 0
    for inf in info.copy():
        if inf["info_type"]=="force":
            pop_info(inf)
            text = text + '\tF' + str(n)+"-"
            force_type = inf.pop('force_type')
            if force_type=="ramp":
                text = text + write_ramp(inf)
            elif force_type=="gaussian":
                text = text + write_gaussian(inf)
            elif force_type=="zero":
                text = text + write_zero(inf)
            info.remove(inf)
            n += 1
            
    text = text + "\nCONSTRAINTS\n"
    n = 0
    for inf in info.copy():
        if inf["info_type"]=="constraint":
            pop_info(inf)
            text = text + '\tC' + str(n)+"-"
            cons_type = inf["constraint"].pop('type')
            if cons_type=="fixed":
                text = text + write_fixed(inf)
            elif cons_type=="contact":
                text = text + write_contact(inf)
            elif cons_type=="surface_contact":
                text = text + write_surface_contact(inf)
            info.remove(inf)
            n += 1
            
    text = text + "\nINITIAL CONDITIONS\n"
    n = 0
    for inf in info.copy():
        if inf["info_type"]=="initial":
            pop_info(inf)
            text = text + '\tI' + str(n)+"-"
            ini_type = inf['initial'].pop('type')
            if ini_type=="rest":
                text = text + write_rest(inf)
            info.remove(inf)
            n += 1
    
    with open(write_dir+'/info.txt', 'w', encoding='utf-8') as f:
        f.write(text)


def save_wav(n, save_dir='./results', idx = 0):
    M, K, C, A, b, W, V, Wc, Vc, phi, phi_c, Fext, Fext_phys, x, xd ,xdd, Fc,\
        Fc_phys, h = load_simulation(n, save_dir)
    Fs = int(np.round(1/h)) 
    audio_data = xd[:,idx]
    full_name = save_dir + "/simulation_" + str(n) + "/xd_" + str(idx) + ".wav"
    write(full_name, Fs, audio_data)
    print("Saved as "+full_name)
    
    