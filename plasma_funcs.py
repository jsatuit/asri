#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculates plasma parameters

Regner ut plasmaparametre, herunder:
debyelengde (debye)



Created on Tue Feb  5 09:04:03 2019

@author: jst072
"""

#import konstanter
from konstanter import kB,e,eps,me,amu
from numpy import sqrt

def debye(N,T):
    """Regner ut debyelengde:
        
    INNDATA:
    N - Partikkeltetthet [m⁻³], kan være np.array eller tall
    T - Temperatur til partikkeltypen [K], kan være np.array eller tall, men må ha samme størrelse som N

    
    UTDATA:
    D - Debyelengde til partikkeltypen [m] - samme størrelse som T og N
    
    EKSEMPEL:
    >>> debye(3e11,2000)
    0.00563452181039586
        """
    #Debyelengde
    D = sqrt(eps*kB*T/(N*e**2)) # m
    return D

def wp(N,q=-e,m=me):
    """Regner ut plasmafrekvens. wp = \sqrt(N*qe²/\epsilon*m)
    
    INNDATA
    N - Partikkeltetthet [m⁻³], kan være np.array eller skalar
    q - ladninga til én partikkel [C], skalar. Standard: elektronladning q_e = -1,6022*10⁻¹⁹ C
    m - Masse til én partikkel [kg], skalar. Standard: elektronmasse m_e = 9,1094*10⁻³¹ kg
    
    UTDATA
    wp - plasmafrekvens [rad/s], samme størrelse som N
    
    EKSEMPEL:
    >>> wp(3e11)    
    30900300.978470266
        """
        
    wp = sqrt(N*q**2/(eps*m))
    return wp
def vth(T,m):
    """
    Regner ut termisk hastighet for bestemt ioneslag
    
    INNDATA
    m - partikkelmasse [kg], kan være np.array eller tall
    T - Temperatur til partikkeltypen [K], kan være tall eller np.array, men må da ha samme størrelse som m
    
    UTDATA
    vt - termisk hastighet [m/s], samme størrelser som m
    
    EKSEMPEL
    >>> vth(T = 300,m=16*amu)
    394.8441959421232
    """
    vt = sqrt(kB*T/m)
    return vt
    
def _test(skriv_ut=False):
    import doctest
    doctest.testmod(verbose=skriv_ut)
if __name__ == '__main__':
    Te = 2000 #K
    Ne = 3e11 #K
    D = debye(Ne,Te)
    print(D)
    _test(skriv_ut=True)
