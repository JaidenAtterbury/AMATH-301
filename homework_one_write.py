# Name: Jaiden Atterbury
# AMATH 301
# Homework 1: Write-up problems

import utils  # noqa: F401, do not remove if using a Mac
import numpy as np
import matplotlib.pyplot as plt

# Write-up problems:

# Problem 1:
# The following code will create a straight line. Copy it, save the plot, and
# add it to your writeup
x = np.arange(-5, 5+0.5, 0.5)
y = x
plt.figure(1)
plt.plot(x, y, color="black")
plt.plot(x, y, color="blue", marker="o")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# Explain the connection between the blue circles in the plot and the black
# curve (e.g., how are they related?).

# Problem 2:
# Create another figure similar to the previous figure except representing the
# function f(x) = x^2. Add a title to the figure explaining what is being
# plotted.
x = np.arange(-5, 5+0.5, 0.5)
y = x ** 2
plt.figure(2)
plt.plot(x, y, color="black")
plt.plot(x, y, color="blue", marker="o")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
