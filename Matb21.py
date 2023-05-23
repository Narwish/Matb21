#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 7 19:29:06 2022

@author: Björn Follin, Lucas Early
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import scipy.optimize as optimize



#Task 1
def rie_sum(f, a, b, n):
    
    rie_list = []
    
    while a <= b:
        
        rie_list.append(f(a) * n)
        
        a = a + n
        
    return sum(rie_list)

def f(x):
    
    return 2*x + x**2

def F(x):
    
    return x**2 + (x**3)/3

def difference_simple_integral(a, b, n):
    
    difference = []
    
    while n > 0.0000001:
        
        difference.append(abs(rie_sum(f, a, b, n) - F(b)))
        n = n - 0.0001        
    
    plt.figure()
    plt.title('Difference between simple integral and Riemann')
    plt.xlabel('Partitions')
    plt.ylabel('Difference')
    plt.plot(difference)
    
difference_simple_integral(0, 10, 0.1)

def question(t):
    
    x_deriv = 2*t
    
    y_deriv = 3*(t**2)
    
    return np.sqrt((x_deriv**2) + (y_deriv**2)) 
    
def scipy_int(f, a, b):
    
    answer = integrate.quad(f, a, b)
    
    return max(answer)

#print(scipy_int(question, -2, 1))

#print(rie_sum(question, -2, 1, 0.00001))



n = 0.1

comparison = []

while n > 0.000000001:
   
    riemann = (rie_sum(question,-2,1,n))
   
    scipy = (scipy_int(question,-2,1))
   
    comparison.append(abs(riemann - scipy))
   
    n = n - 0.0001

plt.figure()
plt.plot(comparison)
plt.title('Riemann and Scipy integrate comparison')
plt.xlabel('Partitions')
plt.ylabel('Difference')


#task 2

def g(x, y):
    return 8*x*y - 4*(x**2)*y - 2*x*(y**2) + (x**2)*(y**2)

vectorized = np.vectorize(g)    

x, y = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200))

def h(params):
    x, y = params
    return 8*x*y - 4*(x**2)*y - 2*x*(y**2) + (x**2)*(y**2)

def h_negative(params):
    x, y = params
    return (8*x*y - 4*(x**2)*y - 2*x*(y**2) + (x**2)*(y**2))*(-1)



steps = []

def save_step(k):
    global steps
    steps.append(k)



guess = [1, 1]

minimum = (optimize.fmin(h, guess, callback = save_step))
maximum = (optimize.fmin(h_negative, guess, callback = save_step))


#print(minimum)
#print(maximum)


plt.figure()
plt.contourf(x, y, vectorized(x, y), 200, cmap='rainbow')
plt.colorbar()
plt.contour(x, y, vectorized(x, y), 20, colors='black')
xs = [step[0] for step in steps]
ys = [step[1] for step in steps]
plt.scatter(xs, ys, c=xs, cmap='magma')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Contour plot of g(x, y)')
plt.xlim([-10, 10])
plt.ylim([-10, 10])


# task 3


def j(x, y, z):
    return x + 2*y + z + np.exp(2*z) -1

def z(x, y):
    return optimize.fsolve(j, 0, args = (x, y))[0]


# def Z(z):
#     x = -1
#     y = -1
#     return x + 2*y + z + np.exp(2*z) - 1



# z_other = optimize.fsolve(Z, 0)

h = 0.001

def f_1(x, y):
    return (z(x + h, y) - z(x, y - h))/2*h

def f_2(x, y):
    return (z(x, y + h) - z(x, y - h))/2*h

f_11 = (f_1(h,0)-f_1(-h,0))/2*h

f_12 = (f_1(0,h)-f_1(0,-h))/2*h

f_22 = (f_2(0,h)-f_2(0,-h))/2*h

def P_2(x, y):
    P = z(0,0) + f_1(0, 0)*x + f_2(0,0)*y + (f_22/2)*(x**2) + f_12*x*y + (f_22/2)*(y**2)
    return P

def e(x, y):
    E = abs(z(x, y) - P_2(x, y))
    return E


xvals = []
yvals = []
zvals = []
P_2_vals = []
err_vals = []
 
for n in range(100):
    for m in range(100):
        xvals.append((n-50)/50)
        yvals.append((m-50)/50)


for n in range (10000):
    zvals.append(z(xvals[n], yvals[n])) 
    P_2_vals.append(z(xvals[n], yvals[n]))
    err_vals.append(e(xvals[n], yvals[n]))


xvals = np.array(xvals)
yvals = np.array(yvals)
zvals = np.array(zvals)
P_2_vals = np.array(P_2_vals)
err_vals = np.array(err_vals)


plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(xvals, yvals, zvals)
ax.set_title('z(x,y)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(xvals, yvals, P_2_vals)
ax.set_title('P_2(x,y)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(xvals, yvals, err_vals)
ax.set_title('e(x, y)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
