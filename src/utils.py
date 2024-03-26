# -*- coding: utf-8 -*-

import numpy as np

def give_I_rectangle(w, h):
    return w*h**3/12

def make_grid(x, y):
    x,y = np.meshgrid(x,y,indexing='ij')
    x,y = x.flatten(), y.flatten()
    return x,y
    