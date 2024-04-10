# -*- coding: utf-8 -*-

import numpy as np

def give_I_rectangle(w, h):
    return w*h**3/12

def make_grid(x, y):
    x,y = np.meshgrid(x,y,indexing='ij')
    x,y = x.flatten(), y.flatten()
    return x,y
    
def newton(g, gp, x0, args=None):
    diff    = g(x0, args)
    step = 0
    while diff > 1e-12 and step < 10000:
        x0  -= g(x0, args)/gp(x0, args)
        diff = g(x0, args)
        step += 1
    return x0, diff
        