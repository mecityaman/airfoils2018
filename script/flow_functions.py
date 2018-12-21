# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 12:01:12 2017

@author: Mecit Yaman
"""

import numpy as np
import sympy
from sympy.abc import x, y

def uniform_flow(u_,v_):
    r       = sympy.sqrt(x**2 + y**2)
    theta   = sympy.atan2(y, x)
    return u_*r*sympy.sin(theta)-v_*r*sympy.cos(theta)

def vortex(x_,y_, t_, l_):
    # define a symbolic function for a vortex flow
    # x_, y_ locations
    # vortex intensity
    # l_ vortex panel length
    
    r       = sympy.sqrt((x-x_)**2 + ((y-y_)**2))
    theta   = sympy.atan2(y-y_, x-x_)
    return (t_*l_)/(1*sympy.pi)*sympy.log(r)

def vortex_velocities_lambda(psi):
    # lambdify the symbolic function ie. make it a Pyhton parametric function 

    u = sympy.lambdify((x, y), psi.diff(y), 'numpy')
    v = sympy.lambdify((x, y), -psi.diff(x), 'numpy')
    return u, v

def vortices(vortices_, free_flow=(1.0, 0.0)):
    
    psi = uniform_flow(*free_flow)  
    for x_,y_,t_,l_,__ in vortices_:
        psi += vortex(x_,y_,t_,l_)
    u = sympy.lambdify((x, y), psi.diff(y), 'numpy')
    v = sympy.lambdify((x, y), -psi.diff(x), 'numpy')
    return u, v
