#!/bin/python3
from athena_read import *
import numpy as np
import matplotlib.pyplot as plt
import os

def test(arg, opt=False):
    print('I was given', arg, 'with', opt)

def open_file(filename, **kwargs):
    data = athdf(filename)

def xy_plot(filename, **kwargs):
    filepath = filename[0:-6]+'.png'
    skip = kwargs.get('check_skip', False)
    if skip:
        if os.path.exists(filepath):
            print('I am skipping', filepath)
            return 0

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

    plt.savefig(filepath)

def log_plot(filename, **kwargs):
    filepath = filename[0:-6]+'.png'
    skip = kwargs.get('check_skip', False)
    if skip:
        if os.path.exists(filepath):
            print('I am skipping', filepath)
            return 0

    data = athdf(filename)
    plt.rcParams['figure.figsize'] = [15, 12]
    plt.rcParams['figure.dpi'] = 100
    fig = plt.figure()
    ax = plt.gca()

    X,Y = np.meshgrid(data['x1v'], data['x2v'])

    mesh = plt.pcolormesh(X, Y, np.log(data['rho'][int(len(data['rho'])/2)),:,:], shading="auto", vmin = -1.,
                      cmap="viridis")
    ax.set_aspect('equal')
    plt.colorbar()

    plt.savefig(filepath)


def zoom_plot(filename, **kwargs):
    filepath = filename[0:-6]+'.png'
    skip = kwargs.get('check_skip', False)
    if skip:
        if os.path.exists(filepath):
            print('I am skipping', filepath)
            return 0

    data = athdf(filename)
    plt.rcParams['figure.figsize'] = [15, 12]
    plt.rcParams['figure.dpi'] = 100
    fig = plt.figure()
    ax = plt.gca()

    center = int(len(data['x1v'])/2)
    X,Y = np.meshgrid(data['x1v'][center-5:center+5], data['x2v'][center-5:center+5])

    mesh = plt.pcolormesh(X, Y, data['rho'][int(len(data['rho'])/2),:,:], shading="auto", vmin = -1.,
                      cmap="viridis")
    ax.set_aspect('equal')
    plt.colorbar()

    plt.savefig(filepath)

