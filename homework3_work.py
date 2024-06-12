# Name: Jaiden Atterbury
# AMATH 301
# Homework 3: Coding problems

import numpy as np
import scipy.integrate as scint

# Coding problems:

# Problem 1:
M = np.genfromtxt('Plutonium.csv', delimiter=',')
xdata = M[0, :]
ydata = M[1, :]

# Part (a): Save the step size, h, to the variable A1.
h = xdata[1] - xdata[0]
A1 = h

# Part (b): Use the first-order forward-difference formula to approximate the
# derivative at time t = 0. Save the result to the variable A2.
fd = 1/h * (ydata[1] - ydata[0])
A2 = fd

# Part (c): Use the first-order backward-difference formula to approximate the
# derivative at time t = 40. Save the result to the variable A3.
bd = 1/h * (ydata[40] - ydata[39])
A3 = bd

# Part (d): Use the second-order forward-difference formula to to approximate
# the derivative at time t = 0. Save the result to the variable A4.
sofd = 1/2*h * (-3*ydata[0] + 4*ydata[1] - ydata[2])
A4 = sofd

# Part (e): Use the method you derived to approximate at t = 40. Save your
# answer to the variable A5.
sobd = 1/2*h * (3*ydata[40] - 4*ydata[39] + ydata[38])
A5 = sobd

# Part (f): Combine the results from A4 and A5 along with the second-order
# central-difference scheme on the interior to approximate the derivative
# at all 41 times in t. Save the resulting array to the variable A6.
fp_centered = np.zeros(ydata.shape)
fp_centered[0] = sofd
for k in range(1, len(xdata)-1):
    fp_centered[k] = 1/(2*h)*(ydata[k+1] - ydata[k-1])
fp_centered[-1] = sobd
A6 = fp_centered

# Part (g): The decay rate of Plutonium-239 at a time t is given by
# −1/P * dP/dt Use your result from A6 to estimate the decay rate at all 41
# times in t. Save your result to the variable A7.
decay_rate = (-1 / ydata) * fp_centered
A7 = decay_rate

# Part (h): To get the true decay rate we average over the 41 times. Save the
# average (mean) of these decay rates to the variable A8.
decay_avg = sum(decay_rate) / len(decay_rate)
A8 = decay_avg

# Part (i): If λ is the average decay rate that you found in part (h), then
# the half life of Plutonium-239 is given by the formula ln(2)/λ. Calculate
# the half life, and save it to the variable A9.
half_life = np.log(2) / decay_avg
A9 = half_life

# Part (j): Derive a finite-difference scheme for the second derivative and
# apply it to calculate the second derivative at t = 22. Save the result to
# the variable A10.
sec_cent_deriv = 1/h**2 * (ydata[21] - 2*ydata[22] + ydata[23])
A10 = sec_cent_deriv

# Problem 2
coef = 1 / np.sqrt(2 * np.pi * (8.3) ** 2)
f = lambda x: coef * np.exp((-(x-85) ** 2) / (137.78))
h_values = np.logspace(-1, -16, 16, base=2.0)

# Part (a): Use scipy.integrate.quad function to calculate the “true” value of
# S. Save your answer to the variable A11.
true_value = scint.quad(f, 110, 130)[0]
A11 = true_value

# Part (b): Use the left-sided rectangle rule to approximate S with step sizes
# of h = 2^−1,2^−2,...,2^−16. Save the approximations in an array with 16
# elements and save that to the variable A12.
approx_left = np.zeros(16)
for index, h in enumerate(h_values):
    x_vals = np.arange(110, 130 + h, h)
    y_vals = f(x_vals)
    approx_left[index] = h * sum(y_vals[:-1])
A12 = approx_left

# Part (c): Repeat part (b) with the right-sided rectangle rule. Save the
# result to the variable A13.
approx_right = np.zeros(16)
for index, h in enumerate(h_values):
    x_vals = np.arange(110, 130 + h, h)
    y_vals = f(x_vals)
    approx_right[index] = h * sum(y_vals[1:])
A13 = approx_right

# Part (d): Repeat part (b) with the midpoint rule. Save the result to the
# variable A14.
approx_mid = np.zeros(16)
for index, h in enumerate(h_values):
    x_vals = np.arange(110, 130, h) + h/2
    y_vals = f(x_vals)
    approx_mid[index] = h * sum(y_vals)
A14 = approx_mid

# Part (e): Repeat part (b) with the trapezoidal rule. Save the result to the
# variable A15.
approx_trap = np.zeros(16)
for index, h in enumerate(h_values):
    x_vals = np.arange(110, 130 + h, h)
    y_vals = f(x_vals)
    approx_trap[index] = h/2 * (y_vals[0] + 2 * sum(y_vals[1:-1]) + y_vals[-1])
A15 = approx_trap

# Part (f): Repeat part (b) with Simpson’s rule. Save the result to the
# variable A16.
approx_simp = np.zeros(16)
for index, h in enumerate(h_values):
    x_vals = np.arange(110, 130 + h, h)
    y_vals = f(x_vals)
    approx_simp[index] = h/3 * (y_vals[0] + 4 * sum(y_vals[1:-1:2]) + 2 * sum(y_vals[2:-2:2]) + y_vals[-1])
A16 = approx_simp
