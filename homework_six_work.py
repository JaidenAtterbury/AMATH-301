# Name: Jaiden Atterbury
# AMATH 301
# Homework 6: Coding problems

import numpy as np
import scipy.optimize
import scipy.integrate

# Coding problems:

# Problem 1:

# Part (a): Forward Euler:

# Define the ODE
ode = lambda p: p*(1-p)*(p-1/2)

# Define dt
dt = 0.5

# Define the initial condition
p_0 = 0.9

# Define the t values for getting solution
t = np.arange(0, 10 + dt, dt)

# Define empty version of solution
p = np.zeros(len(t))

# Assign initial condition to solution
p[0] = p_0

# Calculate solution at all points in t
for k in range(len(t) - 1):
    p[k+1] = p[k] + dt * ode(p[k])

A1 = p

# Part (b): Backward Euler:

# Define empty version of solution
p2 = np.zeros(len(t))

# Assign initial condition to solution
p2[0] = p_0

# Calculate solution at all points in t
for k in range(len(t) - 1):
    g = lambda z: z - p2[k] - dt*ode(z)
    p2[k+1] = scipy.optimize.fsolve(g, p2[k])

A2 = p2

# Part (c): Midpoint Method:

# Define empty version of solution
p3 = np.zeros(len(t))

# Assign initial condition to solution
p3[0] = p_0

# Calculate solution at all points in t
for k in range(len(t) - 1):
    k1 = ode(p3[k])
    k2 = ode(p3[k] + 0.5*dt*k1)
    p3[k+1] = p3[k] + dt*k2

A3 = p3

# Part (d): RK4:

# Define empty version of solution
p4 = np.zeros(len(t))

# Assign initial condition to solution
p4[0] = p_0

# Calculate solution at all points in t
for k in range(len(t) - 1):
    k1 = ode(p4[k])
    k2 = ode(p4[k] + 0.5*dt*k1)
    k3 = ode(p4[k] + 0.5*dt*k2)
    k4 = ode(p4[k] + dt*k3)
    p4[k+1] = p4[k] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

A4 = p4

# Problem 2:

# Part (a):

# Define constants:
a = 1/2

# Define ODEs
rprime = lambda R, J: a*R + J
jprime = lambda R, J: -R - a*J

# Define ODE vector:
odefun = lambda t, x: np.array([rprime(x[0], x[1]), jprime(x[0], x[1])])

# Define initial condition:
x_0 = np.array([2, 1])

# Find solution using scipy.integrate.solve_ivp:
sol = scipy.integrate.solve_ivp(odefun, [0, 20], x_0)

# Find individual solutions:
tsol = sol.t
xsol = sol.y

# Find R and J solutions:
rsol = xsol[0, :]
jsol = xsol[1, :]
A5 = rsol
A6 = jsol

# Part (b):

# Find solutions R(20) and J(20)
r20 = rsol[-1]
j20 = jsol[-1]
A7 = np.array([r20, j20])

# Part (c):

# Define dt
step = 0.1

# Define the t values for getting solution
ts = np.arange(0, 20 + step, step)

# Define empty version of solution
x_vec = np.zeros([len(ts), len(x_0)])

# Assign initial condition to solution
x_vec[0, :] = x_0

# Calculate solution at all points in t
for k in range(len(ts) - 1):
    x_vec[k+1] = np.array([x_vec[k, 0] + step * rprime(x_vec[k, 0], x_vec[k, 1]),
                           x_vec[k, 1] + step * jprime(x_vec[k, 0], x_vec[k, 1])])

# Individual Method:
r_vec = np.zeros(len(ts))
r_vec[0] = x_0[0]
j_vec = np.zeros(len(ts))
j_vec[0] = x_0[1]
for k in range(len(ts) - 1):
    r_vec[k+1] = r_vec[k] + step * rprime(r_vec[k], j_vec[k])
    j_vec[k+1] = j_vec[k] + step * jprime(r_vec[k], j_vec[k])

# Extract individual solutions from x_vec
fd_rsol = x_vec[:, 0]
fd_jsol = x_vec[:, 1]

A8 = fd_rsol
A9 = fd_jsol

# Part (d):
fd_r20 = fd_rsol[-1]
fd_j20 = fd_jsol[-1]
A10 = np.array([fd_r20, fd_j20])

# Part (e):
diff_error = np.array([fd_r20, fd_j20]) - np.array([r20, j20])
norm_error = np.linalg.norm(diff_error)
A11 = norm_error

# Problem 3:

# Part (a):

# Define the parameters we are going to use:
g = 9.8
L = 11
sigma = 0.12

# Define the anonymous function we will solve:
tprime = lambda theta, v: v
vprime = lambda theta, v: (-g/L) * np.sin(theta) - sigma * v

# Define the initial condition:
tv_0 = np.array([-np.pi/8, -0.1])

# Part (b):

# Define the ODE vector to be solved:
odefun2 = lambda t, p: np.array([tprime(p[0], p[1]), vprime(p[0], p[1])])
func_value = odefun2(1, np.array([2, 3]))
A12 = func_value

# Part (c):

# Find solution to IVP:
sol = scipy.integrate.solve_ivp(odefun2, [0, 50], tv_0)

# Find individual solutions:
tsol = sol.t
tvsol = sol.y
A13 = tvsol