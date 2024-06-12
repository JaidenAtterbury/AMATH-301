# Name: Jaiden Atterbury
# AMATH 301
# Homework 2: Coding problems

import numpy as np

# Coding problems:

# Problem 1:
square_num = np.array([])
for n in range(1, 33):
    square_num = np.append(square_num, (2 * n ** 3 + 3 * n ** 2 + n) / 6)
A1 = square_num

# Problem 2:
# Part (a):
y1 = 0
term = 0.1
for k in range(100000):
    y1 += term
A2 = y1

y2 = 0
y3 = 0
y4 = 0

term1 = 0.1
term2 = 0.25
term3 = 0.5

for k in range(100000000):
    y2 += term1
    y3 += term2
    y4 += term3

A3 = y2
A4 = y3
A5 = y4

# Part (b):
x1 = abs(10000 - y1)
x2 = abs(y2 - 10000000)
x3 = abs(25000000 - y3)
x4 = abs(y4 - 50000000)

A6 = x1
A7 = x2
A8 = x3
A9 = x4

# Problem 3:
Fibonacci = np.zeros(200)
Fibonacci[0:2] = 1, 1
max_num = 0
for num in range(2, 200):
    fib_num = Fibonacci[num - 1] + Fibonacci[num - 2]
    if fib_num < 1000000:
        Fibonacci[num] = fib_num
    else:
        max_num = num
        break

A10 = Fibonacci
A11 = max_num - 1
A12 = Fibonacci[:max_num]

# Problem 4:
# Part (a):
x = np.linspace(-1 * np.pi, np.pi, 100)
A13 = x

# Part (b):
y = np.zeros(100)
for k in range(4):
    y += ((-1) ** k / np.math.factorial(2 * k)) * (x ** (2 * k))

A14 = y
