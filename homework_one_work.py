
# Name: Jaiden Atterbury
# AMATH 301
# Homework 1: Coding problems

import numpy as np

# Coding problems:

# Problem 1:
# (a) Define x = 3.1 and save it to the variable A1.
x = 3.1
A1 = x

# (b) Define y = −29 and save it to the variable A2.
y = -29
A2 = y

# (c) Define z = 9e (where e is the number such that ln(e) = 1) and save it to
# the variable A3.
z = 9 * np.e
A3 = z

# (d) Define w = e^4 (where e is the number such that ln(e) = 1) and save it to
# the variable A4.
w = np.exp(4)
A4 = w

# (e) Compute sin(π) in python using numpy. Save the result to the variable A5.
A5 = np.sin(np.pi)

# Problem 2:
# Create/calculate the array [cos(0),cos(π/4),cos(π/2),cos(3π/4),cos(π)] and
# save the result to the variable A6. (Hint: I have given you code that creates
# the array x= [0,π/4,π/2,3π/4,π], how can you use this to help you solve this
# problem?
x = np.linspace(0, 1, 5)
x = np.pi * x
x = np.cos(x)
A6 = x

# Problem 3:
# Use linspace and arange, in the numpy package, to create the vectors uand v
# where uis the vector with 6 evenly spaced elements beginning with 3 and
# ending with 4 and where v= (0 0.75 1.5 2.25 3.0 3.75).

# (a) Save the array u to the variable A7.
u = np.linspace(3, 4, 6)
A7 = u

# (b) Save the array v to the variable A8.
v = np.arange(0, 4, 0.75)
A8 = v

# (c) Save the array w = v + 2u to the variable A9.
w = v + 2 * u
A9 = w

# (d) Using python commands, create a vector w whose entries are the entries of
# u cubed (i.e., to the third power). Save the result to the variable A10.
w = u ** 3
A10 = w

# (e) Using python commands, create a vector x whose entries are tan(u)+e^v.
# Save the result to the variable A11.
x = np.tan(u) + np.exp(v)
A11 = x

# (f) Use indexing to save the 3rd number in u to the variable A12.
# Hint: Remember that indexing begins at 0 for python.
A12 = u[2]

# Problem 4:
# Create an array z with elements that have spacing 1/100 between them and
# that begins at −6 and ends at 3.

# (a) Save the array z to the variable A13.
z = np.linspace(-6, 3, 901)
A13 = z

# (b) Use indexing to create another array that consists of every other entry
# in z. Save that array to the variable A14.
A14 = z[0::2]

# (c) Use indexing to create another array that consists of every third entry
# in z. Save that array to the variable A15.
A15 = z[0::3]

# (d) Use indexing to create an array consisting of the last 5 entries in z.
# Save that array to the variable A16.
A16 = z[-5:]
