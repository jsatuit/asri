#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 17:25:30 2019

@author: jst072
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:27:33 2019

@author: jst072
"""

import numpy as N
from numpy.random import randn
#from numpy.linalg import norm
#from numpy.matlib import repmat
import matplotlib.pyplot as P
#from mpl_toolkits.mplot3d import Axes3D
#from radaro import Radar
from konstanter import c,pi,Timer
i = 1j

def krysskorr(freq=230e6,H=100e3,no=100,nv=20,nz=100,vmax=0.01,L=1000,gml=True):
    """
    Regner ut krysskorrelasjonen mellon spredninga i et plasma av to forskjellige EM-bølger der 
    bølgetallsvektorene danner en vinkel på opp til 2*vmax. Det antas at radaren ligger i et plan.
    
    Calculates the cross-correlation between the scattering in a plasma of two different EM-waves where
    the wave vectors form an angle up to 2*vmax. The radar is assumed to be in a (horizontal) plane
    
    INNDATA:
    freq - radarfrekvens (skalar). Standard: 230 MHz
    H    - høyde til sentrum av plasmaet over radaren [m]. Standars: 100 km
    no   - Antall spredere i plasmavolumet (skalar). Standard: 100
    nv   - Antall vinkler mellom bølgetallsvektorene der krysskorrelasjonen skal beregnes (skalar). Standard: 20
    nz   - Antall målinger av plasmaspredninga. Et høyere tall betyr at det numeriske svaret 
            har lavere usikkerhet, men også at beregningene bir tyngre (skalar). Standard: 100
    vmax - Største vinkel [grader]. Standard 0,01 grader
    L    - Bokslengde [m]. Hvor stort plasmavolumet er i alle retningene (foreløpig skalar). Standard: 1000 m
    
    UTDATA:
    r21  - korrelasjon mellom spredningsmålingene med baselinjeavstand d (Beregna numerisk)
    r21a - korrelasjon mellom spredningsmålingene med baselinjeavstand d (Beregna analytisk)
    d    - baselinjeavstand
    
    """
    # Sprederposisjoner / Scatterer position
    spos = randn(3,no,nz,1)*L + N.array([H/100,H/100,H]).reshape(3,1,1,1)
    
    
    # Bølgetallsvektorlengde / Wave number vector magnitude
    kl = 2*pi*freq/c
    
    # Vinkelforskjell mellom spredningsvektorer
    # Angle difference between scattering vectors
#    ang = N.linspace(0,vmax,nv).reshape(1,1,1,nv)
    ang = N.logspace(N.log10(vmax)-2,N.log10(vmax),nv).reshape(1,1,1,nv)
    # Regner om vinkel til bakkelinjeavstand
    d = 2*H*N.tan(ang.flatten()*pi/180)
#    d = H*N.sin(2*ang)
    
    tpos = N.array([0,0,0]).reshape(3,1,1,1)
    r1pos = N.vstack([d/2,N.zeros((2,nv))]).reshape(3,1,1,nv)
    r2pos = N.vstack([-d/2,N.zeros((2,nv))]).reshape(3,1,1,nv)
    
    
    # Bragg-Spredningsvektorer
#    k2 = kl* (N.array([0,0,-1]).reshape(3,1,1,1)*N.cos(ang*pi/180) + N.array([1,0,1]).reshape(3,1,1,1)*N.sin(ang*pi/180))
#    k1 = kl* (N.array([0,0,-1]).reshape(3,1,1,1)*N.cos(ang*pi/180) - N.array([1,0,1]).reshape(3,1,1,1)*N.sin(ang*pi/180))
    
    ki  = kl * N.array([0,0,1]).reshape(3,1,1,1)
    ks1 = kl * N.vstack([ N.sin(ang.flatten()*pi/180), N.zeros([1,nv]), -N.cos(ang.flatten()*pi/180)]).reshape(3,1,1,nv)
    ks2 = kl * N.vstack([-N.sin(ang.flatten()*pi/180), N.zeros([1,nv]), -N.cos(ang.flatten()*pi/180)]).reshape(3,1,1,nv)
    
    k1 = ks1-ki
    k2 = ks2-ki
    
    
    kx1 = N.squeeze(k1[0,])
    ky1 = N.squeeze(k1[1,])
    kz1 = N.squeeze(k1[2,])
    kx2 = N.squeeze(k2[0,])
    ky2 = N.squeeze(k2[1,])
    kz2 = N.squeeze(k2[2,])
    
    # Utgående tid
    Ti = N.linalg.norm(tpos-spos,axis=0)/c
    # Ankommende tider (dimensjoner no,nz,nv)
    Tr1 = N.linalg.norm(r1pos-spos,axis=0)/c
    Tr2 = N.linalg.norm(r2pos-spos,axis=0)/c
    
    
    
    if gml:
        # d-dimensjoner, S-spredere, M-målinger, V-vinkler
        Z22a = N.einsum('dV,dSM->SMV',N.squeeze(k2),N.squeeze(spos))
        Z2 = N.sum(N.exp(i*Z22a),axis=0) 
    
        Z12a = N.einsum('dV,dSM->SMV',N.squeeze(k1),N.squeeze(spos))
        Z1 = N.sum(N.exp(i*Z12a),axis=0)
    
    # Gjennomsnitt(forventninger) / Averages (means)
        Ez1z1 = N.abs(N.sum(Z1*Z1.conj(),axis=0))
        Ez2z1 = N.abs(N.sum(Z2*Z1.conj(),axis=0))
    # Analytisk beregning
#       Ez2z1a = N.abs(N.exp(H*(kx1-kx2)*i - (L*L*(kx1-kx2)**2 + L*L*(ky1-ky2)**2 + L*L*(kz1-kz2)**2)/2 ))
        Ez2z1a = N.abs(N.exp(H*(kz1-kz2)*i - (L*L*(kx1-kx2)**2 + L*L*(ky1-ky2)**2 + L*L*(kz1-kz2)**2)/2 ))
        Ez2z1b = N.abs(N.exp(              - (L*L*(kx1-kx2)**2)/2 ))
    else:
        Z2 = N.sum(N.exp(i*(2*pi*freq)*(Ti+Tr2)),axis=0)
        Z1 = N.sum(N.exp(i*(2*pi*freq)*(Ti+Tr1)),axis=0)
        
        # Gjennomsnitt(forventninger) / Averages (means)
        Ez1z1 = N.abs(N.sum(Z1*Z1.conj(),axis=0))
        Ez2z1 = N.abs(N.sum(Z2*Z1.conj(),axis=0))
        
#        R1 = H
#        L2x = L*L/(1+4*L*L*(c*(kx1-kx2)**2/(4*pi*freq) - 4*pi*freq/c)**2/R1**2)
#        L2y = L*L/(1+4*L*L*(c*(ky1-ky2)**2/(4*pi*freq) - 4*pi*freq/c)**2/R1**2)
#        L2z = L*L/(1+4*L*L*(c*(kz1-kz2)**2/(4*pi*freq) - 4*pi*freq/c)**2/R1**2)
#        Ez2z1a = N.abs(N.exp( - (L2x**2*(kx1-kx2)**2 + L2y**2*(ky1-ky2)**2 + L2z**2*(kz1-kz2)**2) ))
        
        Ez2z1a = N.abs(N.exp(H*(kz1-kz2)*i - (L*L*(kx1-kx2)**2 + L*L*(ky1-ky2)**2 + L*L*(kz1-kz2)**2)/2 ))

#        n = 1e11
#        k1Lk1 = (L*L*(kx1)**2 + L*L*(ky1)**2 + L*L*(kz1)**2)/2
#        k2Lk2 = (L*L*(kx2)**2 + L*L*(ky2)**2 + L*L*(kz2)**2)/2
#        Ez2z1a = (1+n*N.exp(-k1Lk1/1)*N.exp(-k2Lk2/2)) / N.sqrt( (1+n*N.exp(-k1Lk1))*(1+n*N.exp(-k2Lk2)) )
        
        
#    if N.any(Ez2z1a != Ez2z1b):
#        print('Ikke lik')
#    print(((kz1-kz2)-(kx1-kx2))*H % (2*pi))

    r21 = Ez2z1/Ez1z1
    r21a = Ez2z1a
    
    return r21,r21a,d
def calcboxsize(freq=230e6,H=100e3,no=100,nv=20,nz=100,vmax=0.01,corr=0.95):
    """
    Regner ut maksimal boksstørrelse for ønska korrelasjon og gitt bakkelinjelengde. Det antas at radaren ligger i et plan.
    Calculates the maximal box size for wanted correlations and given baseline length. The radar must be in a (horizontal) plane
    
    INNDATA:
    freq - radar frekvecy (skalar). Standard: 230 MHz
    H    - height of the center of the plasma over the radar [m]. Standard: 100 km
    no   - Number of scatterers in the plasma volume (skalar). Standard: 100
    nv   - Number of angles between the wave vectors where the cross-correlation shall be calculated (skalar). Standard: 20
    nz   - Number of measurements of the plasma scattering. A larger number reduces the uncertainty of the numerical value, but also that the computation becomes more  
            heavy (skalar). Standard: 100
    vmax - Larget angle[grader]. Standard 0,01 grader
    corr - Wanted korrelasjon. Standard = 95 %
    
    UTDATA:
    L    - Største boksstørrelse der korrelasjonen er over ønska verdi
    d    - bakkeavstand
    """
    # Bølgetallsvektorlengde / Wave number vector magnitude
    kl = 2*pi*freq/c
    
    # Vinkelforskjell mellom spredningsvektorer
    # Angle difference between scattering vectors
    ang = N.logspace(-2,N.log10(vmax),nv).reshape(1,nv)
    # Regner om vinkel til bakkelinjeavstand
    d = 2*H*N.tan(ang*pi/180)
    
    # Bragg-Spredningsvektorer
#    k2 = kl* (N.array([0,0,-1]).reshape(3,1,1,1)*N.cos(ang*pi/180) + N.array([1,0,1]).reshape(3,1,1,1)*N.sin(ang*pi/180))
#    k1 = kl* (N.array([0,0,-1]).reshape(3,1,1,1)*N.cos(ang*pi/180) - N.array([1,0,1]).reshape(3,1,1,1)*N.sin(ang*pi/180))
    
    ki  = kl * N.array([0,0,1]).reshape(3,1)
    ks1 = kl * N.vstack([ N.sin(ang*pi/180), N.zeros([1,nv]), -N.cos(ang*pi/180)]).reshape(3,nv)
    ks2 = kl * N.vstack([-N.sin(ang*pi/180), N.zeros([1,nv]), -N.cos(ang*pi/180)]).reshape(3,nv)
    
    k1 = ks1-ki
    k2 = ks2-ki
    
    
    kx1 = k1[0,].reshape(1,nv)
    ky1 = k1[1,].reshape(1,nv)
#    kz1 = N.squeeze(k1[2,])
    kx2 = k2[0,].reshape(1,nv)
    ky2 = k2[1,].reshape(1,nv)
#    kz2 = N.squeeze(k2[2,])
    
#    L = 2*N.sqrt( N.log(1/corr**2) / ((kx1-kx2)**2+(ky1-ky2)**2) )
    L = 2*N.sqrt( N.log(1/corr**2) / (kx1-kx2)**2 )
    return L,d
# Høyde height
#H = N.array([1e3,20e3,50e3,100e3,500e3,3000e3])
H = N.array([100e3])

# Antall spredere
# # of scatterers
no = 1000
# Antall vinkler
# # of angles
nv = 10
# Antall målinger av Z (For å få gjennomsnitt)
# of measurements of Z (to get average)
nz = 100

# Største vinkel / largest angle
angmax = N.arctan(0.01)*180/pi
# Boksstørrelser / Box sizes
#L=N.linspace(0,900,20)
L = N.logspace(0,N.log10(900),80)
## Radar
freq = 230e6
#freq = 50e6

# Preallokering
r21  = N.zeros([len(L),nv,len(H)])
r21a = N.zeros([len(L),nv,len(H)])
d    = N.zeros([len(H),nv])
with Timer('Beregninger'):
    for i2 in range(len(H)):
        for i1 in range(len(L)):    
            r21[i1,:,i2],r21a[i1,:,i2],d[i2,] = krysskorr(freq,H[i2],no,nv,nz,N.arctan(1500/H[i2])*180/pi,L[i1]/2,gml=False)
            print("runde",i1+1,"av",len(L))

# Plots cross-correlations between baselines
for i2 in range(len(H)):
    f = P.figure(figsize=(10,8))
    
    for i1 in range(nv):
        P.scatter(L,r21[:,i1,i2],marker='o',label=None)
#        P.plot(L,r21a[:,i1,i2],label=(str(int(round((d[i2,i1]))))))
#        P.plot(L,r21a[:,i1,i2],label=str(round(d[i2,i1],3)))
        P.semilogx(L,r21a[:,i1,i2],label=(str(int(round(d[i2,i1])))+' m'))
    P.semilogx(L,0.95*N.ones(L.shape),'k',label='95 % correlation')

    P.xlabel('Feature size [m]')
    P.ylabel('Correlation')
    P.title(('Cross-correlation between baselines. (Range '+str(int(round(H[i2]/1e3)))+" km)"))
    P.legend(title=('Baseline for h='+str(int(round(H[i2]/1e3)))+" km"))
    P.ylim(0,1.05)
    P.xlim(0,L.max())
    P.show()
#    f.savefig(("Crosscorrelation_normdist.pdf"))#+str(int(H/1e3))+"m.pdf"))
    f.savefig(("Crosscorrelation_normdist_"+str(int(H[i2]/1e3))+"km.pdf"))
# Plots the relative difference between analytically and numerically calculated cross-correlation (not needed)
"""for i2 in range(len(H)):
    f = P.figure(figsize=(10,8))    
    for i1 in range(nv):
        P.semilogx(L,r21[:,i1,i2]-r21a[:,i1,i2],label=(str(int(round(d[i2,i1])))+' m'))
    P.xlabel('Feature size [m]')
    P.ylabel('Correlation deviation from analytical solution')
    P.title(('Cross-correlation deviation between baselines. (Range '+str(int(round(H[i2]/1e3)))+" km)"))
    P.legend(title=('Baseline for h='+str(int(round(H[i2]/1e3)))+" km"))
    P.ylim(-0.1,0.1)
    P.xlim(0,L.max())
    P.show()"""
# Ønska korrelasjoner
# Correlations we look for
corr = 0.95
corr2= 0.05
f1 = P.figure(figsize=(7.5,5))
ax1 = f1.gca()
f2 = P.figure(figsize=(7.5,5))
ax2 = f2.gca()
for h in N.linspace(100e3,600e3,6):
    Ls,ds = calcboxsize(freq,h,no,nv,nz,angmax*100e3/h,corr)
    
    # Radarsynsfelt (1 grads åpning)
#    rad = h*N.sin(1*pi/360)
#    # Oppløsning/antall bokser i radarstrålen i ei retning
#    res = rad/Ls
    
    ax1.loglog(ds.T,Ls.T,label=str(int(h/1e3))+' km')
#    ax2.plot(ds.T,res.T,label=str(int(h/1e3))+' km')

#Bakkeavstander / Baseline lengths
Lb = N.linspace(1,1000,1000)
dr = (c/freq)*H[0]/(pi*Lb) * N.sqrt(N.log(1/corr**2))
dr2 = (c/freq)*H[0]/(pi*Lb) * N.sqrt(N.log(1/corr2**2))
#res2 = pi*dr*N.sin(1*pi/360) / ((c/freq)*N.sqrt((1+dr**2/(4*H[0]**2))*N.log(1/corr**2)))
#ax2.plot(ds.T,res.T)
ax2.loglog(Lb.T,dr.T,label='largest baseline for cross-correlation ' + str(int(corr*100))+' %')
ax2.loglog(Lb.T,dr2.T,label='largest baseline for cross-correlation ' + str(int(corr2*100))+' %')
ax2.loglog(Lb.T,75*N.ones(Lb.T.shape),label='EISCAT3D core max baseline')
ax2.loglog(Lb.T,1257*N.ones(Lb.T.shape),label='EISCAT3D outlier max baseline')



f1.suptitle(('Largest feature size for cross-correlation '+str(int(corr*100))+' %'),y=0.92)
f2.suptitle(('Largest baseline for crosscorrelation '+str(int(corr*100))+' %'),y=0.92)
ax1.set_xlabel('Largest baseline [m]')
ax2.set_xlabel('Feature size [m]')

ax1.set_ylabel('Largest feature size [m]')
ax2.set_ylabel('Longest baseline [m]')
ax1.grid(True)
ax2.grid(True)
#P.xlim(0,d.max())
ax1.legend(title='Height')
ax2.legend()
P.show()
#f1.savefig(("Boxsize_normdist.pdf"))
f2.savefig(("Resolution_normdist.pdf"))


