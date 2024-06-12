# Name: Jaiden Atterbury
# AMATH 301
# Homework 8: Coding problems

import numpy as np
import cv2

# Coding problems:

# Problem 1:

# Part (a):
rotation_z = lambda theta: np.array([[np.cos(theta), -np.sin(theta), 0],
                                     [np.sin(theta), np.cos(theta), 0],
                                     [0, 0, 1]])

A = rotation_z(np.pi/4)

A1 = A

# Part (b):
b1 = np.array([[3],
               [np.pi],
               [4]])

x1 = np.linalg.solve(A, b1)

A2 = x1

# Problem 2:

# Part (b):
# Define W8, W9, W10, and W11
W8 = 12000
W9 = 9200
W10 = 9000
W11 = 19200

# Predefine the square-root term and the matrix, A, and the vector b
s = 1 / np.sqrt(17)
AA = np.zeros([19, 19])
b2 = np.zeros([19, 1])

# Equation 1
j = 1-1
AA[j, 1-1] = -s
AA[j, 2-1] = 1
AA[j, 12-1] = s

# Equation 2
j = 2-1
AA[j, 1-1] = -4 * s
AA[j, 12-1] = -4 * s

# Equation 3
j = 3-1
AA[j, 2-1] = -1
AA[j, 3-1] = 1
AA[j, 13-1] = -s
AA[j, 14-1] = s

# Equation 4
j = 4-1
AA[j, 13-1] = -4 * s
AA[j, 14-1] = -4 * s

# Equation 5
j = 5-1
AA[j, 3-1] = -1
AA[j, 4-1] = 1
AA[j, 15-1] = -s
AA[j, 16-1] = s

# Equation 6
j = 6-1
AA[j, 15-1] = -4 * s
AA[j, 16-1] = -4 * s

# Equation 7
j = 7-1
AA[j, 4-1] = -1
AA[j, 5-1] = 1
AA[j, 17-1] = -s
AA[j, 18-1] = s

# Equation 8
j = 8-1
AA[j, 17-1] = -4 * s
AA[j, 18-1] = -4 * s

# Equation 9
j = 9-1
AA[j, 5-1] = -1
AA[j, 6-1] = s
AA[j, 19-1] = -s

# Equation 10
j = 10-1
AA[j, 6-1] = -4 * s
AA[j, 19-1] = -4 * s

# Equation 11
j = 11-1
AA[j, 6-1] = -s
AA[j, 7-1] = -1

# Equation 12
j = 12-1
AA[j, 7-1] = 1
AA[j, 8-1] = -1
AA[j, 18-1] = -s
AA[j, 19-1] = s

# Equation 13
# This is the first equation that has a W_n term in it. That does not multiply
# any of the unknowns, W_k, so we move it to the right-hand side: it is part of
# b
j = 13-1
AA[j, 18-1] = 4 * s
AA[j, 19-1] = 4 * s
b2[j] = W8

# Equation 14
j = 14-1
AA[j, 8-1] = 1
AA[j, 9-1] = -1
AA[j, 16-1] = -s
AA[j, 17-1] = s

# Equation 15
j = 15-1
AA[j, 16-1] = 4 * s
AA[j, 17-1] = 4 * s
b2[j] = W9

# Equation 16
j = 16-1
AA[j, 9-1] = 1
AA[j, 10-1] = -1
AA[j, 14-1] = -s
AA[j, 15-1] = s

# Equation 17
j = 17-1
AA[j, 14-1] = 4 * s
AA[j, 15-1] = 4 * s
b2[j] = W10

# Equation 18
j = 18-1
AA[j, 10-1] = 1
AA[j, 11-1] = -1
AA[j, 12-1] = -s
AA[j, 13-1] = s

# Equation 19
j = 19-1
AA[j, 12-1] = 4 * s
AA[j, 13-1] = 4 * s
b2[j] = W11

# Part (c):
A3 = AA

# Part (d):
x2 = np.linalg.solve(AA, b2)

A4 = x2

# Part (e):
xabs = np.abs(x2)
xmax = np.amax(xabs)

A5 = xmax

# Part (f):
collapse_weight = 0
j_val = 0
for k in range(10000):
    b2[16] += 5
    forces = np.linalg.solve(AA, b2)
    forces_abs = np.abs(forces)
    if np.amax(forces_abs) > 44000:
        collapse_weight = b2[16, 0]
        arg_max = np.argmax(forces_abs)
        for index, force in enumerate(forces_abs):
            if force > 44000:
                j_val = index + 1
                break
        break

A6 = collapse_weight
A7 = j_val

# Problem 3:

# Part (a):
Aimg = cv2.imread('olive.jpg', 0)
pix_num = Aimg.shape[0] * Aimg.shape[1]
A8 = pix_num

# Part (b):
U, S, Vt = np.linalg.svd(Aimg, full_matrices=False)
A9 = U
A10 = S
A11 = Vt.T

# Part (c):
fifteen_svs = S[0:15]
A12 = fifteen_svs

# Part (d):
total_energy = np.sum(S)
largest_sv = fifteen_svs[0]
A13 = largest_sv / total_energy

# Part (e):
largest15_sv = sum(fifteen_svs)
A14 = largest15_sv / total_energy

# Part (f):
r = 0
for k in range(1, len(S)):
    sum_energy = sum(S[0:k])
    ratio_energy = sum_energy / total_energy
    if ratio_energy >= 0.75:
        r = k
        break
A15 = r
