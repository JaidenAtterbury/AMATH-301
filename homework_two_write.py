# Name: Jaiden Atterbury
# AMATH 301
# Homework 2: Write-up Problems

import utils  # noqa: F401, do not remove if using a Mac
import numpy as np
import matplotlib.pyplot as plt

# Write-up Problems:

# Problem 2


def get_cos_taylor(array, order):
    y_values = np.zeros(100)
    for k in range(order + 1):
        y_values += ((-1) ** k / np.math.factorial(2 * k)) * (array ** (2 * k))
    return y_values


x = np.linspace(-1 * np.pi, np.pi, 100)

# Part (i):
y = np.cos(x)
plt.plot(x, y, color="k", linewidth=2)

# Part (ii):
y = get_cos_taylor(x, 1)
plt.plot(x, y, color="b", linestyle="--", linewidth=2)

# Part (iii):
y = get_cos_taylor(x, 3)
plt.plot(x, y, color="r", linestyle="-.", linewidth=2)

# Part (iv):
y = get_cos_taylor(x, 14)
plt.plot(x, y, color="m", linestyle=":", linewidth=2)

# Part (v):
plt.xlabel("x-values", fontsize=14)

# Part (vi):
plt.ylabel("cos(x) approximations", fontsize=14)

# Part (vii):
plt.title("cos(x) and its Taylor approximations", fontsize=14)

# Part (viii):
plt.legend(("cos(x)", "n = 1 Taylor", "n = 3 Taylor", "n = 14 Taylor"))

plt.show()
