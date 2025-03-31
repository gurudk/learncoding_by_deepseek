import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display

def pi(x, V):
    '''
    Density of the target distribution, up to a constant.

    x -- np array of size k
    V -- np array of size k*k
    '''
    return np.exp(-.5 * np.dot(x, np.dot(V, x)))


def prop(x):
    '''
    Random proposition for the Metropolis-Hastings algorithm.
    Uses the Random Walk Metropolis formula with unit variance.

    x -- np array of size k
    '''
    return x + normal(size=len(x))


def q(x, y):
    '''
    Probability density of transition x to y, up to a constant.
    Uses the Random Walk Metropolis formula with unit variance.

    x -- np array of size k
    y -- np array of size k
    '''
    dist = x - y
    return np.exp(-.5 * np.dot(dist, dist))

example_V = np.array([[5.,4.5],[4.5,5.]])
display(example_V)
display(np.linalg.inv(example_V))

d1 = np.linalg.inv(example_V)[1,0]/np.linalg.inv(example_V)[0,0]

x = np.arange(-3.0, 3.0, 0.1)
y = np.arange(-3.0, 3.0, 0.1)
X, Y = np.meshgrid(x, y)
Z = np.array([[pi(np.array([X[i,j],Y[i,j]]),V=example_V) for j in range(len(X[0]))] for i in range(len(X))])
fig1 = plt.figure()
CS = plt.contour(X, Y, Z)
plt.clabel(CS, inline=1, fontsize=10)
plt.show()