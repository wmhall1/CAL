#!/bin/python3
from athena_read import *
import numpy as np
import matplotlib.pyplot as plt


def test(arg, opt=False):
    print('I was given', arg, 'with', opt)

def open_file(filename, **kwargs):
    data = athdf(filename)

def xy_plot(filename, **kwargs):
    data = athdf(filename)
    plt.rcParams['figure.figsize'] = [15, 12]
    plt.rcParams['figure.dpi'] = 100  
    fig = plt.figure()
    ax = plt.gca()

    X,Y = np.meshgrid(data['x1v'], data['x2v'])

    mesh = plt.pcolormesh(X, Y, data['rho'][int(len(data['rho'])/2),:,:], shading="auto", vmin = -1.,
                      cmap="viridis")
    ax.set_aspect('equal')
    plt.colorbar()

    plt.savefig(filename[0:-6]+'.png')


