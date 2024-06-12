# Name: Jaiden Atterbury
# AMATH 301
# Homework 4: Coding problems

import numpy as np
import scipy.integrate
import scipy.optimize

# Coding problems:

# Problem 1:
x = lambda t: 11 / 6 * np.exp(-t / 12) - 11 / 6 * np.exp(-t)

# Part (a):
xprime = lambda t: -11 / 72 * np.exp(-t / 12) + 11 / 6 * np.exp(-t)
A1 = xprime(1.5)

hiv_root = scipy.optimize.fsolve(xprime, 1.5)[0]
A2 = hiv_root

hiv_max = x(hiv_root)
A3 = hiv_max

# Part (b):
neg_x = lambda t: -11 / 6 * np.exp(-t / 12) + 11 / 6 * np.exp(-t)
argmin_of_negx = scipy.optimize.fminbound(neg_x, 0, 10)
min_of_x = x(argmin_of_negx)
A4 = np.array([argmin_of_negx, min_of_x])

# Problem 2:

# Part (a):
f = lambda x, y: (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
fp = lambda p: f(p[0], p[1])

f_value = fp(np.array([3, 4]))
A5 = f_value

# Part (b):
f_argmin = scipy.optimize.fmin(fp, np.array([-3, -2]))
A6 = f_argmin

# Part (c):
gradf_xy = lambda x, y: np.array([4*x**3 - 42*x + 4*x*y + 2 * y ** 2 - 14,
                                  4*y**3 - 26*y + 4*x*y + 2 * x ** 2 - 22])
gradf_p = lambda p: gradf_xy(p[0], p[1])

del_f_argmin = gradf_p(f_argmin)
A7 = del_f_argmin

two_norm_argmin = np.linalg.norm(del_f_argmin)
A8 = two_norm_argmin

# Part (d):
p = np.array([-3, -2])
tol = 10 ** -7
iterations = 0
for k in range(2000):
    grad = gradf_p(p)
    if abs(np.linalg.norm(grad)) < tol:
        break
    phi = lambda t: p - t * grad
    f_of_phi = lambda t: fp(phi(t))
    tstar = scipy.optimize.fminbound(f_of_phi, 0, 1)
    p = phi(tstar)
    iterations += 1

# Part (e):
A9 = p
A10 = iterations
