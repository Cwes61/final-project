# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 07:54:53 2019

@author: jzbfa1
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from __main__ import h
from __main__ import q_x
from __main__ import q_y
from __main__ import k
from __main__ import N

n = 0.30
dx = 1

def C_new(C, Node, dx, i, j):
    if Node == 0:
        # interior cell
        C_n = (2*C[i-1,j] + 2*C[i,j-1] - Vx*((C[i+1, j] - C[i-1,j])/2*dx) -Vy((C[i,j+1] - C[i,j-1])/2*dx))/((4/dx**2) + k[i,j])
    elif Node == 1:
        C_n = C[i,j]
    elif Node == 21:
        # no flux left
        C_n = (2*C[i-1,j] + 2*C[i,j-1] - Vx*((C[i+1, j] - C[i-1,j])/2*dx))/((4/dx) + k[i,j])
    elif Node == 22:
        # no flux right
        C_n = (2*C[i-1,j] + 2*C[i,j-1] - Vx*((C[i-1, j] - C[i+1,j])/2*dx))/((4/dx) + k[i,j])
    elif Node == 23:
        # no flux bottom
        C_n = (2*C[i-1,j] + 2*C[i,j-1] - Vy*((C[i,j-1] - C[i,j+1])/2*dx))/((4/dx) + k[i,j])
    elif Node == 24:
        # no flux top
        C_n = (2*C[i-1,j] + 2*C[i,j-1] - Vy*((C[i,j+1] - C[i,j-1])/2*dx))/((4/dx) + k[i,j])
    else:
        C_n = -99
    return C_n

@jit
def FinDiff(C, Node, dx, tol):
    err = tol + 1
    iters = 0
    lam = 1.8
    Nx, Ny = np.shape(C)
    while err > tol and iters < 10000:
        err = 0
        iters += 1
        for j in range(Ny):
            for i in range(Nx):
                C_o = C[i,j]
                C_n = C_new(C, Node[i,j], dx, i, j)
                C_n = lam*C_n + (1-lam)*C_o
                err = max(err, abs(C_n-C_o))
                C[i,j] = C_n
    return C, iters

#Set up nodes
Con = np.zeros_like((N), dtype=int)
Con[1:-1, 1:-1] = 0 #Interior
Con[:, 0] = 1 #Left
Con[:, -1] = 1 #Right
Con[0,:] = 23 #Bottom
Con[-1,:] = 24 #Top


#Initilize Vx and Vy
Vx = np.zeros_like(q_x)
Vy = np.zeros_like(q_y)
for i in range(207):
    Vx[i,:] = q_x[i,:]/n
for j in range(103):
    Vy[:,j] = q_y[:,j]/n


#initial guess for head
C = np.ones_like(Con, dtype='float')*5
C[0,:] = 10 #C is 10 along the left
C[-1,0:6] = np.arange(6) #C = z along the right
C[np.where(N==-1)] = -99 #C doesn't exisit above the free surface

#run the finite difference code
C, iters = FinDiff(C, Con, 1, 1e-2)

# Take out the -99 values for plotting purposes and replace with 0
C[np.where(C==-99)]=np.nan
plt.matshow(np.transpose(h), origin='lower')
plt.colorbar()
plt.quiver(np.linspace(0, 200, 201)[::spc], np.linspace(0, 100, 101)[::spc], np.transpose(q_x[::spc, ::spc]), np.transpose(q_y[::spc, ::spc]))














