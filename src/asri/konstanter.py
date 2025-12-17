#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from numpy import pi
import time

# Permittivity
eps = 8.854e-12 #C²/Jm
#Permeability
my = 4*pi*1e-7 #Ns²/C²
#Elektron mass
me = 9.1094e-31 # kg
#Boltzmanns constant
kB = 1.3807e-23 #J/K
#Elementary charge
e = 1.6022e-19 #C
#Speed of light
c = 299792458 #m/s
#"electron radius"
re = my*e**2/(4*pi*me)
#atom mass unit
amu = 1.6605390e-27 #kg



class MatrixDimensionsDoesNotAgree(Exception):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message
        
        
        
