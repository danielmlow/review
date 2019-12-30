#!/usr/bin/env python3

'''
Author: Daniel M. Low
license: Apache 2.0

'''

#
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
from random import randint, seed

seed(1)


if __name__ == "__main__":
    format = 'eps'
    output_dir = 'data/outputs/'
    plt.clf()
    plt.style.use('seaborn-ticks')
    x = np.arange(10)
    y = [(5*n)+ randint(1,15) for n in x]
    # fit line
    coefs1 = poly.polyfit(x, y, 1)
    x_new1 = np.linspace(x[0], x[-1], 50)
    ffit1 = poly.polyval(x_new1, coefs1)

    # fit polynomial
    coefs = poly.polyfit(x, y, 9)
    x_new = np.linspace(x[0], x[-1], 50)
    ffit = poly.polyval(x_new, coefs)

    # plot
    plt.plot(x_new, ffit, '-', c='darkorange')
    plt.plot(x_new1, ffit1, '-', c='k')
    plt.plot(x,y,'.', c='dodgerblue', markersize=14, alpha=None)
    plt.xlabel('Speech inputs',fontsize=18)
    plt.ylabel('Disorder severity',fontsize=18)
    plt.tick_params(
        axis='both',
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        left=False,
        right=False,
        labelleft=False) # labels along the bottom edge are off
    plt.savefig(output_dir+'box1.'+format, format=format, dpi=1000)

