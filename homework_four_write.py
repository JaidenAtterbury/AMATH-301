# Name: Jaiden Atterbury
# AMATH 301
# Homework 4: Writeup problems
import utils  # noqa: F401, do not remove if using a Mac
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate
import scipy.optimize

# Writeup problems:

# Problem 1:

# Part (a):
# i.
f = lambda x, y: (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

# ii.
x = np.linspace(-7, 7, 40)
y = np.linspace(-7, 7, 40)
X, Y = np.meshgrid(x, y)

# iii.
fig = plt.figure()
ax = plt.axes(projection='3d')

