# -*- coding: utf-8 -*-

'''
This file generates a wav file from simulation data.
'''

# LIBRARIES -------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import os

os.chdir(os.getcwd()+'/..')

print(os.getcwd())

import src.UK as UK
import src.data as data
import src.disp as disp


# MAIN PROGRAM ----------------------------------------------------------------

n   = 4
idx = 312

data.save_wav(n, idx=idx)