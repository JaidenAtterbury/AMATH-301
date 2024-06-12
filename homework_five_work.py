# Name: Jaiden Atterbury
# AMATH 301
# Homework 5: Coding problems

import numpy as np
import scipy.optimize

# Coding problems:

# Problem 1:

# Part (a):
carbon_data = np.genfromtxt("CO2_data.csv", delimiter=",")
year = carbon_data[:, 0]
CO2 = carbon_data[:, 1]

A1 = year
A2 = CO2

# Part (b):
def sumSquaredError(a, b, r):
    y = lambda t: a + b * np.exp(r * t)
    error = np.sum((y(year) - CO2) ** 2)
    return error

A3 = sumSquaredError(300, 30, 0.03)

# Part (c):
sse_adapter = lambda p: sumSquaredError(p[0], p[1], p[2])
init_guess = np.array([300, 30, 0.03])
sse_optimal = scipy.optimize.fmin(sse_adapter, init_guess)

A4 = sse_optimal

# Part (d):
optimal_error = sse_adapter(sse_optimal)

A5 = optimal_error

# Part (e):
def maxError(a, b, r):
    y = lambda t: a + b * np.exp(r * t)
    error = np.amax(np.abs(y(year) - CO2))
    return error

max_adapter = lambda p: maxError(p[0], p[1], p[2])

A6 = max_adapter(init_guess)

max_optimal = scipy.optimize.fmin(max_adapter, init_guess, maxiter=2000)

A7 = max_optimal

# Part (f):
def seasonal_sse(a, b, r, c, d, e):
    y = lambda t: a + b * np.exp(r * t) + c * np.sin(d * (t - e))
    error = np.sum((y(year) - CO2) ** 2)
    return error

seas_sse_adapt = lambda p: seasonal_sse(p[0], p[1], p[2], p[3], p[4], p[5])

init_guess_2 = np.array([300, 30, 0.03, -5, 4, 0])

A8 = seas_sse_adapt(init_guess_2)

# Part (g):
opt_init_guess = np.append(sse_optimal, init_guess_2[3:6])

seas_optimal = scipy.optimize.fmin(seas_sse_adapt, opt_init_guess, maxiter=2000)

A9 = seas_optimal
print(A4)
print(A9)

# Part (h):
seas_error = seas_sse_adapt(seas_optimal)

A10 = seas_error

# Problem 2:

# Part (a):
salmon_data = np.genfromtxt('salmon_data.csv', delimiter=',')

year = salmon_data[:,0]
salmon = salmon_data[:,1]

# Part (b):
line_best = np.polyfit(year, salmon, 1)

A11 = line_best

# Part (c):
cubic_best = np.polyfit(year, salmon, 3)

A12 = cubic_best

# Part (d):
quintic_best = np.polyfit(year, salmon, 5)

A13 = quintic_best

# Part (e):
exact = 752638

err1 = np.abs(np.polyval(line_best, 2022) - exact) / exact
err2 = np.abs(np.polyval(cubic_best, 2022) - exact) / exact
err3 = np.abs(np.polyval(quintic_best, 2022) - exact) / exact

err_array = np.array([err1, err2, err3])

A14 = err_array
