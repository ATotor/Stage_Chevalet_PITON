# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def set_gui_qt():
    try:
        import IPython
        shell = IPython.get_ipython()
        shell.enable_matplotlib(gui='qt')
    except:
        pass  