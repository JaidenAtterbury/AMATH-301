# Name: Jaiden Atterbury
# AMATH 301
# Homework 7: Coding problems

import numpy as np
import scipy.optimize
import scipy.integrate

# Coding problems:

# Problem 1: Adam's Moulton Method

# Part (a):
dydt = lambda t, y: 5e5 * (-y + np.sin(t))
ts = np.linspace(0, 2*np.pi, 100)
dt = ts[1] - ts[0]
y = np.zeros(len(ts))
y0 = 0
y[0] = y0

g = lambda z: z - y0 - dt*(dydt(ts[0], y0) + dydt(ts[1], z)) / 2
A1 = g(3)

# Part (b):

for k in range(len(y) - 1):
    g = lambda z: z - y[k] - dt*(dydt(ts[k], y[k]) + dydt(ts[k+1], z)) / 2
    y[k+1] = scipy.optimize.fsolve(g, y[k])

A2 = y

# Problem 2: Belousov-Zhabotinsky (BZ) reaction:

# Part (a):

# Define constants:
s = 77.27
w = 0.161
q = 1

# Define system of ODEs:
y1_prime = lambda y1, y2, y3: s * (y2 - y1*y2 + y1 - q*y1**2)
y2_prime = lambda y1, y2, y3: (-y2 - y1*y2 + y3)/s
y3_prime = lambda y1, y2, y3: w * (y1 - y3)

# Create adapter ODE:
odefun = lambda t, y: np.array([y1_prime(y[0], y[1], y[2]),
                                y2_prime(y[0], y[1], y[2]),
                                y3_prime(y[0], y[1], y[2])])

A3 = odefun(1, np.array([2, 3, 4]))

# Part (b):

# Create qs vector:
qs = np.logspace(0, -5, 10)

# Create initial conditions
y_init = np.array([1, 2, 3])

# Create solution template:
exp_sol = np.zeros([3, 10])

# Define function that redifnes ODEs corresponding to value of q
def get_ODEs(qval):
    y1_prime2 = lambda y1, y2, y3: s * (y2 - y1*y2 + y1 - qval*y1**2)
    y2_prime2 = lambda y1, y2, y3: (-y2 - y1*y2 + y3)/s
    y3_prime2 = lambda y1, y2, y3: w * (y1 - y3)


    odefun2 = lambda t, y: np.array([y1_prime2(y[0], y[1], y[2]),
                                    y2_prime2(y[0], y[1], y[2]),
                                    y3_prime2(y[0], y[1], y[2])])

    return odefun2

# Find the solution at the last time for 10 different q values (RK45):
for index, qval in enumerate(qs):
    ode_sys = get_ODEs(qval)
    sol = scipy.integrate.solve_ivp(ode_sys, [0, 30], y_init, t_eval=[30])
    yarr = np.array([sol.y[0, 0], sol.y[1, 0], sol.y[2, 0]])
    exp_sol[:,index] = yarr

A4 = exp_sol

# Part (c):

# Create solution template:
imp_sol = np.zeros([3, 10])

# Find the solution at the last time for 10 different q values (BDF):
for index, qval in enumerate(qs):
    ode_sys = get_ODEs(qval)
    sol = scipy.integrate.solve_ivp(ode_sys, [0, 30], y_init, t_eval=[30], method="BDF")
    yarr = np.array([sol.y[0, 0], sol.y[1, 0], sol.y[2, 0]])
    imp_sol[:,index] = yarr

A5 = imp_sol

# Problem 3: Van der Pol oscillator:

# Part (a):

# Define the parameters we are going to use:
mu = 200

# Define the ODEs:
dxdt = lambda x, y: y
dydt = lambda x, y: mu*(1 - x**2)*y - x

A6 = dxdt(2, 3)
A7 = dydt(2, 3)

# Part (b):

# Define adapter ODE:
odefun = lambda t, v: np.array([dxdt(v[0], v[1]),
                                dydt(v[0], v[1])])

# Define t value array:
tspan = [0, 400]

# Define initial conditions
x_init = np.array([2, 0])

# Get RK45 solution:
sol_45 = scipy.integrate.solve_ivp(odefun, tspan, x_init)
A8 = sol_45.y[0, :]

# Part (c):

# Get BDF solution:
sol_bdf = scipy.integrate.solve_ivp(odefun, tspan, x_init, method="BDF")
A9 = sol_bdf.y[0, :]

# Part (d):
len_45 = len(sol_45.t)
len_BDF = len(sol_bdf.t)

A10 = len_45 / len_BDF

# Part (e):
dxdt = lambda x, y: y
dydt = lambda x, y: mu*y - x

A11 = dxdt(2, 3)
A12 = dydt(2, 3)

# Part (f):
A = np.array([[0, 1], [-1, mu]])
A13 = A

# Part (g):

# Define dt:
dt = 0.01

# Define tspan:
tspan = np.arange(0, 400 + dt, dt)

# Part (h):

# Define solution template:
ans = np.zeros([len(tspan), 2])
ans[0, :] = x_init


for k in range(len(tspan) - 1):
    ans[k+1, :] = ans[k, :] + dt * A@ans[k, :]

A14 = ans[: ,0]

# Part (i):

# Define the identity matrix:
I = np.eye(2)

# Define C:
C = I - dt * A
A15 = C

# Define solution template:
ans_back = np.zeros([len(tspan), 2])
ans_back[0, :] = x_init

# Backward-Euler solution:
for k in range(len(tspan) - 1):
    ans_back[k+1, :] = np.linalg.solve(C, ans_back[k, :])

A16 = ans_back[: ,0]
