#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from numpy import array,meshgrid,linspace,zeros,ones,pi,exp,logical_and,cos,ndarray,vstack,float32

def make_gaussian_blobs(a=zeros([1000,1000]),centres=array([(380,500),(200,800),(800,200)]).T,sd=array((50,50,50)),mul = None):
#    print(a.shape,centres,sd)
    if a.ndim != 2:
        raise ValueError('a må ha to dimensjoner (være ei matrise)')
    if centres.ndim != 2:
        raise ValueError('centres må ha to dimensjoner (være ei matrise)')
    if sd.ndim != 2:
        raise ValueError('sd må ha to dimensjoner (være ei matrise)')
    x,y = meshgrid(linspace(1,a.shape[0],a.shape[0]),linspace(1,a.shape[1],a.shape[1]))    
    
    if mul is None:
        mul = ones(sd.shape)
    
    for i1 in range(centres.shape[1]):
        x0,y0 = centres[:,i1]
        
        ledd1 = -(x-x0)**2/(2*sd[:,i1]**2)
        ledd2 = -(y-y0)**2/(2*sd[:,i1]**2)
        eksponent =  ledd1 + ledd2
        nevner = (2*pi*sd[:,i1]**2)
        a += mul[:,i1] * exp(eksponent) / nevner
    return a
def cossq2d(x,y,b=1,x0=0,y0=0,normalizer=True):
    """
    Cos²(b*pi/2(x-x0)), |x-x0|<b
    0,                  ellers
    
    UTDATA
    z - funksjonen evaluert i x og y. Har samme dimensjoner som x og y
    """
    if not (isinstance(x,ndarray) and isinstance(y,ndarray)):
        raise TypeError('Både x og y må være matriser/vektorer!')
    if not (x.shape == y.shape):
        raise ValueError('Vektorene/matrisene x og y må ha samme størrelse!')
    if not isinstance(b,(int,float)):
        raise TypeError('b må være et enkelt tall')
    
    innafor = logical_and(abs(x-x0) < b , abs(y-y0) < b)
    z = zeros(x.shape)
    z[innafor] = cos(b*pi/(2*b*b)*(x[innafor]-x0))**2 * cos(b*pi/(2*b*b)*(y[innafor]-y0))**2
    if normalizer:
        z /= b*b
    return z

def makecos2blobs(A,sentre,bredder,bilde,normaliser=True):
    """
    Tegner cos²-bobler i et bilde
    
    
    """
    if not isinstance(A,ndarray):
        raise TypeError('A må være ei matrise!')
    if not A.ndim == 2:
        raise TypeError('Matrisa A må være todimensjonal!')
    if not len(sentre) == 2:
        raise TypeError('Lista med sentre må være todimensjonal/inneholde to lister for x- og y-koordinatene til sentrene')
    if not isinstance(bredder,(int,float)):
        raise TypeError('Breddene må være like og være ett tall')
    
    sx = sentre[0]
    sy = sentre[1]
    
    x,y = meshgrid(linspace(1/2,A.shape[0]-1/2,A.shape[0]),linspace(1/2,A.shape[1]-1/2,A.shape[1]))
    
    for i1 in range(len(sx)):
        A += cossq2d(x,y,bredder,sx[i1],sy[i1],normaliser) * bilde[i1]
    return A
def compimag(A,B,quiet=True,norm=False):
    """
    Sammenligner to bilder med midlere kvadraters avvik
    Compares two images with mean square error
    
    INNDATA:
    A - bilde
    B - bilde med samme dimensjoner og størrelse som A
    quiet - Oppgir om skrivestrenger (printf) ikke skal vises. Standard: True
    ########norm - Oppgir om bildene skal normaliseres
    """
    a = A.shape
    b = B.shape
    
    
    if a != b:
        if not quiet:
            print(a,b)
#        raise ValueError('Bildene A og B må være like store!')
        if A.shape[0]/A.shape[1] == B.shape[0]/B.shape[1]:
            if A.size < B.size:
                C=A
                A=B
                B=C
                
            forstørrelse = A.shape[0]/B.shape[0]  
            sx,sy=meshgrid(forstørrelse*linspace(1+1/2,B.shape[0]+1/2,B.shape[0]) - forstørrelse/2,\
                           forstørrelse*linspace(1+1/2,B.shape[1]+1/2,B.shape[1]) - forstørrelse/2)
            sentre = vstack([sx.reshape(1,-1),sy.reshape(1,-1  )])
            B = makecos2blobs(zeros(A.shape),sentre,forstørrelse,B.reshape(-1,1),False)
#            else:
#                raise ValueError('Bildene A og B må være like store!')
        else:
            raise ValueError('Bildene A og B må være like store!')
    if not quiet:
        print('Power of A',A.sum(),'Power of B',B.sum())
    if norm:
        A = A/abs(A.max())
        B = B/abs(B.max())
        
    if A.ndim == 2:
        MSE = sum(sum((A-B)**2))/A.size
    elif A.ndim == 1:
        MSE = sum((A-B)**2)/A.size
    
    return MSE

def diffcomb(Tx,Rx=None):
    """
    Regner ut alle mulighetene med forskjellen mellom Tx og Rx (= Txi-Rxj for alle i og j). Tx (og Rx) må være vektorer (endimensjonale).
    """
    if Rx is None:
        Rx = Tx.copy()
    
    dx1,dx2 = meshgrid(Tx,Rx)
    dx = float32(dx1-dx2).reshape(1,-1)
    del(dx1,dx2)
  
    return dx

def uniktol(a,ac,ai=None,ai2=None,tol = 0.01):
    """
    Finner hvor ofte en vektor gjentas i ei matrise innafor en viss toleranse.
    
    
    """
    # Unike vektorer
    au = []
    # Antall like vektorer
    ant = []
    
    if ai is None:
        pass
    else:
        # For invertering av vektorene, slik at man får den lange opprinnelige matrisa tilbake
        ainv = ai.copy()
    if not ai2 is None:
        # Hvis man ønsker å gjenskape vektorkomprimeringa
        afrem = []
    
    # Hvor mange vektorer det er opprinnelig (i a)
    a_len = a.shape[1]
    
    for i1 in range(a_len):
        
        if i1%100 == 0:
            print(i1+1,'of',a_len)
        
        found = False
        indeks = None
        # Ser etter om vektoren allerede eksisterer i funnmatrisa
        for i2 in range(len(au)):
            
            if all((au[i2] - a[:,i1])**2 < (tol*ones(au[i2].shape))**2):
                found = True
                indeks = i2
#                print(au[i2],' = ',a[:,i1],au[i2] - a[:,i1],tol*ones(au[i2].shape) )
                break
        # Fant den ikke, legger den til funnmatrisa
        if not found:
            au.append(a[:,i1])
            ant.append(ac[i1])
            if not ai is None:
                ainv[ainv==i1] = len(ant)-1
            if not ai2 is None:
                afrem.append(ai2[i1])
        # Fant vektoren, øker antallet like med én
        else:
            ant[indeks] += ac[i1]
            if not ai is None:
                ainv[ainv==i1] = indeks

    # Omformer til matriser
    au = array(au).T
    ant = array(ant)
    if ai is None:
        out =  [au,ant]
    else:
        out =  [au,ainv,ant]
    
    if not ai2 is None:
        out.append(afrem)
    return out
