import numpy as np
from math import *
from spg_overlay.utils.utils import normalize_angle
from matplotlib import pyplot as plt


def f(x,u):
    f1 = x[0]+u[0]*cos(x[2]+u[1])
    f2 = x[1]+u[0]*sin(x[2]+u[1])
    f3 = normalize_angle(x[2]+u[2]) # +2*u[2]
    return np.array([f1,f2,f3])

def F_k(x,u):
    return np.array([
            [1, 0,-u[0]*sin(x[2]+u[1])],
            [0, 0,u[0]*sin(x[2]+u[1])],
            [0, 0,1],
        ])

 