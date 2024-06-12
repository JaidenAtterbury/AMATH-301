# Name: Jaiden Atterbury
# AMATH 301
# Homework 3: Write-up Problems

import utils  # noqa: F401, do not remove if using a Mac
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scint

# Write-up Problems:

# Problem 1:
M = np.genfromtxt('Plutonium.csv', delimiter=',')
xdata = M[0, :]
ydata = M[1, :]

h = xdata[1] - xdata[0]
sofd = 1/2*h * (-3*ydata[0] + 4*ydata[1] - ydata[2])
sobd = 1/2*h * (3*ydata[40] - 4*ydata[39] + ydata[38])

fp_centered = np.zeros(ydata.shape)
fp_centered[0] = sofd
for k in range(1, len(xdata)-1):
    fp_centered[k] = 1/(2*h)*(ydata[k+1] - ydata[k-1])
fp_centered[-1] = sobd

# Part (a): Load in the data and plot the data as black markers with black
# lines in between (’-ok’). Add axis labels, legends, and a title.
plt.plot(xdata, ydata, "-ok", label="Plutonium Data")
plt.xlabel("Time (in years)")
plt.ylabel("Amount of Plutonium-239")
plt.title("Plot of the Plutonium Data")
plt.legend()
plt.show()

# Part (b): Create a new figure (you can use plt.figure()). On it, plot the
# derivative that you calculated using the second order methods, A6. Plot this
# using green markers with green lines between the markers (’-og’). Include
# axis labels, legends, and a title.
plt.figure()
plt.plot(xdata, fp_centered, "-og", label="Derivative of the Plutonium Data")
plt.xlabel("Time (in years)")
plt.ylabel(r'$\frac{dP}{dt}$ values')
plt.title("Plot of the Derivative of the Plutonium Data")
plt.legend()
plt.show()

# Problem 2:
coef = 1 / np.sqrt(2 * np.pi * (8.3) ** 2)
f = lambda x: coef * np.exp((-(x-85) ** 2) / (137.78))
h_values = np.logspace(-1, -16, 16, base=2.0)

# True value:
true_value = scint.quad(f, 110, 130)[0]

# Left-hand rule:
approx_left = np.zeros(16)
for index, h in enumerate(h_values):
    x_vals = np.arange(110, 130 + h, h)
    y_vals = f(x_vals)
    approx_left[index] = h * sum(y_vals[:-1])

# Right-hand rule:
approx_right = np.zeros(16)
for index, h in enumerate(h_values):
    x_vals = np.arange(110, 130 + h, h)
    y_vals = f(x_vals)
    approx_right[index] = h * sum(y_vals[1:])

# Midpoint rule:
approx_mid = np.zeros(16)
for index, h in enumerate(h_values):
    x_vals = np.arange(110, 130, h) + h/2
    y_vals = f(x_vals)
    approx_mid[index] = h * sum(y_vals)

# Trapezoidal rule:
approx_trap = np.zeros(16)
for index, h in enumerate(h_values):
    x_vals = np.arange(110, 130 + h, h)
    y_vals = f(x_vals)
    approx_trap[index] = h/2 * (y_vals[0] + 2 * sum(y_vals[1:-1]) + y_vals[-1])

# Simpsons rule:
approx_simp = np.zeros(16)
for index, h in enumerate(h_values):
    x_vals = np.arange(110, 130 + h, h)
    y_vals = f(x_vals)
    approx_simp[index] = h/3 * (y_vals[0] + 4 * sum(y_vals[1:-1:2]) + 2 * sum(y_vals[2:-2:2]) + y_vals[-1])

# Part (a): For each of the five methods used to calculate S, create an array
# of length 16 containing the absolute value of the error for the different
# step sizes.
left_error = np.abs(approx_left - true_value)
right_error = np.abs(approx_right - true_value)
mid_error = np.abs(approx_mid - true_value)
trap_error = np.abs(approx_trap - true_value)
simp_error = np.abs(approx_simp - true_value)

# Part (b): Plot the errors versus the step size for each of the five methods
# using a log-log plot. Use a different color and marker type for each method.
plt.figure()
plt.loglog(h_values, left_error, "ok", label="Left-hand rule")
plt.loglog(h_values, right_error, "*r", label="Right-hand rule")
plt.loglog(h_values, mid_error, "+b", label="Midpoint rule")
plt.loglog(h_values, trap_error, ".g", label="Trapezoidal rule")
plt.loglog(h_values, simp_error, "^y", label="Simpsons rule")

# Part(c):
plt.loglog(h_values, (0.0005) * h_values, "-r", label=r'$O(h)$', linewidth=2)
plt.loglog(h_values, (0.00003) * h_values ** 2, ":b", label=r'$O(h^2)$', linewidth=2)
plt.loglog(h_values[0:10], (0.00000003) * h_values[0:10] ** 4, ".-k", label=r'$O(h^4)$', linewidth=2)
plt.loglog(h_values, np.array([10 ** -16] * 16), "k", label="Machine Precision")
plt.xlabel("h values")
plt.ylabel("Error")
plt.title("Plot of the Errors of the 5 Main Numerical Integration Methods")
plt.legend(loc="bottom right", bbox_to_anchor=(1.00001, 0.325))
plt.show()
