#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from numpy import array,shape,vstack,hstack,repeat,linspace,sin,cos,tan,exp,\
    zeros,loadtxt,sqrt,mod,ones,ndarray,flipud,spacing,float32,\
    complex64,trace,diag,iscomplex,unique,meshgrid,log10,dot,arctan2,\
    multiply,matmul,where,asarray,logical_and,nan
from numpy import sum as asum
from numpy.linalg import norm,inv,LinAlgError,pinv,svd#,lstsq,solve,det
from numpy.matlib import repmat
from numpy.random import standard_normal
#from numpy.fft import fft2,ifft2,fftshift
from konstanter import c,pi,Timer#,MatrixDimensionsDoesNotAgree
from matplotlib.pyplot import plot,scatter,xlabel,ylabel,xlim,ylim,clim,legend,axes,hist,\
    figure,colorbar,title,subplot,semilogy,pcolormesh,gca,grid,contourf,loglog,savefig,close,clf,subplots_adjust
from matplotlib.pyplot import show as show_plots

from imageio import imread

from h5py import File,is_hdf5

# This is the path to where intermedtate files are saved. You will need much space on this hard drive.
# You !have to! change this path and un-comment it
#fillagresti = '/home/your_user/imaging/radaro3/'

# Tikhonov-regulariseringsparametre
trp1 = 10
trp2 = 100
i = 1j

alfabetet = [chr(i) for i in range(ord('a'),ord('{'))]+['æ','ø','å']


# Where the center of the transmitter array is located
nullpkt = hstack([ loadtxt('Eiscat3D_transmitter.txt',skiprows=1).reshape(2,-1,order='F')[:,0], 0]).reshape(3,1)
# Locations of all core subarrays
eiscat3dtpos = loadtxt('e3d_array.txt').T
eiscat3dtpos = vstack([eiscat3dtpos , zeros([ 1,eiscat3dtpos.shape[1] ]) ])[:,0:109:1]
# Locations of the outrigger subarrays
eiscat3drpos = loadtxt('EISCAT3D_receivers.txt',skiprows=1).T.reshape(2,-1,order='F')
eiscat3drpos = vstack([eiscat3drpos , zeros([1,shape(eiscat3drpos)[1] ]) ])-nullpkt

# Some random and testing arrays
mersestjerne = array([[0,10,0],[0,0,0],[-sqrt(3)*5,-5,0],[5*sqrt(3),-5,0]]).T
TE = array([[-10,5,0],[0,5,0],[10,5,0],[0,-15,0]]).T
Tto = array([[-10,6.5,0],[0,3,0],[10,6.5,0],[0,-16,0]]).T
boks = 7.5*array([[-1,1,0],[1,1,0],[-1,-1,0],[1,-1,0]]).T
null = zeros([3,1])

# The forms of the target. If they produce errors, they can be commented, except for j19, storj and prikk129
J = flipud(loadtxt('J.csv')).T
J_liten = loadtxt('J_liten.csv')
prikk = zeros([43,43])
prikk[21,21] = 1
prikk129 = zeros([129,129])
prikk129[64,64] = 1
storprikk = zeros([43,43])
storprikk[20:23,20:23] = 1
prikk45 = zeros([45,45])
prikk45[23,23] = 1
storj = flipud(loadtxt('storj.csv')[4:-4,0:-1]).T
j19 = flipud(loadtxt('j19.csv')).T

# Loads the aurora image, but currently not available
nordlys = flipud(imread("aurora.png")[:-1,:,0]).T/255
from misc import make_gaussian_blobs,compimag,diffcomb,uniktol
from plasma import Plasma

def get_target(im,norm):
    """
    Gir ut ønska bilde,
    Hands out wanted image as a numpy array
    """
    # Går gjennom lista med bildenavn og velger ønska bilde
    if im == 'nordlys':
        target = nordlys#[21:(n+21),21:(n+21)]
        error("Currently the aurora image is not available")
    elif im == 'boble1':
        target = make_gaussian_blobs(zeros((97,97)),array([(80,80)]).T,20*ones((1,1)))
    elif im == 'boble2':
        target = make_gaussian_blobs(zeros((97,97)),array([(80,80)]).T,10*ones((1,1)))
    elif im == 'j19':
        target = j19
    elif im == 'psf':
        target = prikk129
    else: # Hvis ønska navn ikke var i lista
#        print('Advarsel: Fant ikke bildet du ville ha, så du får et nordlysbilde')
        target = nordlys
        print("Didn't find your target. You will get an aurora image")
        
        
    # Normaliserer
    if norm:
        target = target/target.max()
        
    return target


def MF(A,m):
    """
    Tilpassa filter
    Matched filter
    
    A - theory matrix
    m - measurement vector
    
    A - teorimatrise
    m - målevektor
    """
    return A.conj().T@m
def capon(A,m,vi=None,normalize=True):
    """
    capon-filter
    
    A - theory matrix
    m - measurement vector
    vi - inverse comprimation vector. When A and m are comprimated (because many baselines are equal), 
        they have to be decompressed to use capon correctly.
    normalize - This argument is not in use. 
        
    
    A - teorimatrise
    m - målevektor
    vi - invers komprimeringsvektor. Når A og m er komprimerte (fordi mange 
            grunnlinjer er like), må de dekomprimeres for å normalisere målingene. 
            Dersom vi ikke er oppgitt, antas det at målingene er normaliserte!
    normalize - sier om målingene skal mormaliseres. Hvis ikke må fullstendig målevektor oppgis her (Standard: True)
    """
#    if normalize:
#        if not vi is None:
#            # Normalizing
#            # Number of effective receivers
#            effrec = int(sqrt(len(vi)))
#            
#            # Normalizing crosscorrelations
#            m_mat = m[vi].reshape(effrec,-1)
#            mc = m[vi]/(trace(m_mat)/m_mat.shape[0])
#            m = m/(trace(m_mat)/m_mat.shape[0])
#        else:
#            mc = m
#            effrec = int(sqrt(m.size))
#    else:
#        if not isinstance(normalize,ndarray):
#            raise ValueError('Hoppa over')
    if vi is None:
        mc = m.copy()
    else:
        mc = m[vi]
    effrec = int(sqrt(mc.shape[0]))
    
    # Inverts cross-correlations
    W = pinv(mc.reshape(effrec,-1),rcond=0.0003)
    w = W.reshape(-1,1)
    
    # Reconstructs image
    if vi is None:
        B = A.conj().T
    else:
        B = A[vi,:].conj().T 
        
    try:
        xh = 1/(B@w)
    except MemoryError:
        nevner = matmulfor(B,w)
        xh = 1/nevner
    
    return xh
def TSVD(A,m,rcond):
    """
    Truncated singular value decomposition
    
    A - Theory matrix
    m - measurement vector
    rcond - AT which relatice singular value the SVD will be truncated
    
    Trunkert singulærverdidekomponering
    
    A - teorimatrise
    m - målevektor
    rcond- minste tillatte relative singulærverdi
    """
    # Truncated least squares
    AH = pinv(A,rcond=rcond)
    xh = AH@m
    return AH,xh
def SSVD(A,m,rcond):
    """
    This was a test. The function should not be used
    "Hjemmelagd" singulærverdidekomponering
    
    A - teorimatrise
    m - målevektor
    rcond- minste tillatte relative singulærverdi
    """
    # Singular value decomposition
    
    U,S,VH = svd(A)
    plot(S.diagonal())
    title('Singular values')
    
    AH = VH.conj().T@inv(S)@U.conj().T
    xh = AH@m
    return AH,xh
def MKM(A,m):
    """
    Least square method
    Minste kvadraters metode
    A - teorimatrise
    m - målevektor
    """
    s = covA(A)
    AH = s@A.conj().T
    xh = AH@m
    return AH,xh
def covA(A):
    """
    Calculated covariance of A, (A^H*A)⁻¹
    Regner ut kovariansen til A
    """
    AHA = A.conj().T@A
    s = inv(AHA)
    return s
def matmulfor(A,B):
    """
    Matrix multiplication as a for loop. It is very slow, but needs less memory. 
    Gives out C = A*B. Should not be used in first place.
    
    Matrisemultiplikasjon med for-løkke. Tregere, men krever mindre minne
    c = A*B
    """
    
    if A.dtype.name == 'float32' or A.dtype.name == 'float64':
        c = zeros((A.shape[0],B.shape[1]),dtype='float32')
    else:
        c = zeros((A.shape[0],B.shape[1]),dtype='complex64')
    
    for m,rad in enumerate(A):
        for n,kol in enumerate(B.T):
            c[m,n] = sum(rad*kol)
    return c
def normalize_measurements(m,vi=None):
    """
    Normalizes measurements. The original code for this function is currently on an unavailable server.
    Normaliserer målinger
    """
    print("Unfortunately, this function (normalize) does not currently represent its final state and may contain errors.")
    if vi is None:
        # Normaliserer målingene
        effrec = int(sqrt(len(m)))
        m_mat = m.reshape(effrec,-1)
#        mn = m/(trace(m_mat)/m_mat.shape[0])
        mn = m
    else:
        # Number of effective receivers
        effrec = int(sqrt(len(vi)))
        # Normalizing crosscorrelations
        m_mat = m[vi].reshape(effrec,-1)
        mn = m/(trace(m_mat)/m_mat.shape[0])
    return mn
def svddiag(x,plasma,A,m,rcond,vi=None,show=True,normcomp=False,layout='-1',new=False):
    """
    Shows plots for investigating the SVD solution: Original image, singular values, and the solutions LS, TSVD, and Tikhonov (two different)
    x - original image
    plasma - Plasmaobject
    A - theory matrix
    m . measurement vector
    rcond - Where to truncate the singular values
    vi - If compressed theory matrix and measurement vector (Should not be the case)
    show - show the plots
    normcomp - Parameter for calculating the similatiry to the original. (Should normally not be used)
    layout - What layout the radar has. Is used for the filenames then saving intermediate files.
    new - Force new SVD calculations
    
    
    Regner ut intensiteten med (T)SVD, plotter resultatet og diverse tilleggsplott
    
    x - Opprinnelig bilde
    plasma - Plasmaet slik vi trur det er og vil finne ut elektrontettheta til.
    A - teorimatrise til forsøket. Størrelsen gjenspeiler oppløsninga til resultatet
    m - målevektor
    methods - liste med metoder som brukes for å regne ut intensiteten
    methodparams - liste med parametre til hver enkelt metode
    
    """
    
    # Skalerer målingene slik at elektrontettheta blir lik igjen
    m *= plasma.val.size
    
    oppl = int(sqrt(x.shape[0]))
    plasmat = Plasma(plasma.height,plasma.opening,coord='xy',res=(oppl,oppl),val=None,farfield=plasma.farfield,show=False,cut=False)
    
    if show:
        figure(figsize=(20,10))
        subplot(2,3,1)
        plasmat.showestim(sol=x,t='Original image',cmap='jet')
    
    xrec = []
    
    
    ##### Laster inn data
    # (Antatt) plasmaoppløsning
    pr = int(sqrt(plasma.val.size))
    prt = '_ps-'+str(pr)+'x'+str(pr)

    filnavn = fillagresti + 'tminv/L-' + str(layout) + '_' + 'rad-' + str(round(plasma.radius)) + '_hkm-'+str(round(plasma.height/1e3)) + prt
    if not plasma.farfield:
        filnavn += '_nf-'
    else:
        filnavn += '_ff-'
    filnavn += 'svd.hdf5'
    
    if is_hdf5(filnavn) and not new:
        print('Hadde SVD. Laster inn...')
        with File(filnavn, "r") as f:
            U  = asarray(f['U'])
            S  = asarray(f['S'])
            VH = asarray(f['VH'])
    else:
        # Singulkærverdidekomponering
        U,S,VH = svd(A.conj(), full_matrices=False)
        # Lagrer invers matrise til neste gang
        with File(filnavn, "w") as f:
            f.create_dataset("U",  data=U )
            f.create_dataset("S",  data=S )
            f.create_dataset("VH", data=VH )
    # Normaliserte singulærverdier
    Sn = S/S.max()
    # Indeksene til singulærverdiene som er store nok
    ids = Sn>rcond
    # Inverse singulærverdier
    si = 1/S
    # Setter inversen til uønska singulærverdier til null
    si[~ids] = 0
    # Ganger sammen matrisene til pseudoinvers til teorimatrisa
    hs = multiply(si.reshape(-1,1)[range(sum(ids)),:], U[:,range(sum(ids))].T)
#    print(VH.T.shape,hs.shape)
    AH2 = matmul(VH[range(sum(ids)),:].T, hs)
    # Tickhonov-inverterte singulærverdier
    sit = 1/S * S*S/(S*S+trp1*trp1)
    AHT = matmul(VH.T, multiply(sit.reshape(-1,1), U.T))
    # Tickhonov-inverterte singulærverdier
    siu = 1/S * S*S/(S*S+trp2*trp2)
    AHU = matmul(VH.T, multiply(siu.reshape(-1,1), U.T))
    # Tickhonov-inverterte singulærverdier
#    siv = 1/S * S*S/(S*S+100*100)
#    AHV = matmul(VH.T, multiply(siv.reshape(-1,1), U[:,range(sum(ids))].T))
    
    # Gir ut estimat på ukjente
    try:
        AHMKM,xMKM=MKM(A,m)
    except LinAlgError:
        xMKM = zeros((A.T@m).shape)
    xh = AH2@m
    xht= AHT@m
    xhu= AHU@m
#    xhv= AHV@m
#    print(((xh-xh2)**2).sum())
    if show:
        subplot(2,3,2)
        plasma.showestim(xMKM.real,'Least squares','jet')
        subplot(2,3,3)
        
        plasma.showestim(xh.real,'TSVD','jet')
        subplot(2,3,4)
        plasma.showestim(xht.real,'Tikhonov-alfa='+str(trp1),'jet')
        subplot(2,3,5)
        plasma.showestim(xhu.real,'Tikhonov-alfa='+str(trp2),'jet')
    #    subplot(2,3,5)
    #    plasma.showestim(xhv.real,'Tikhonov-alfa=100','jet')
        subplot(2,3,6)
        semilogy(1/S,label='Inverse singular values')
        semilogy(si,label='Truncated inverse singular values')
        semilogy(sit,label='Tikhonov, alfa = '+str(trp1))
        semilogy(siu,label='Tikhonov, alfa = '+str(trp2))
    #    semilogy(siv,label='Tikhonov, alfa = 100')
        xlabel('Value number')
        ylabel('1/value')
        title('Inverse singular values')
        legend(loc='best')
    
    # Ulikheter fra "sanne" verdier
    SANN = x.reshape(oppl,oppl)
    sim = [compimag(SANN,xMKM.real.reshape(int(sqrt(xMKM.shape[0])),-1),norm=normcomp),\
           compimag(SANN,xh.real.reshape(int(sqrt(xh.shape[0])),-1),norm=normcomp),\
           compimag(SANN,xht.real.reshape(int(sqrt(xht.shape[0])),-1),norm=normcomp),\
           compimag(SANN,xhu.real.reshape(int(sqrt(xhu.shape[0])),-1),norm=normcomp)]
    
    xrec.append(xh)
    # Legger til kovarians til TSVD (må ganges med variansen til målingene) og tikhonov med alfa = 1 og 10.
    xrec.append(AH2@AH2.conj().T)
    xrec.append(AHT@AHT.conj().T)
    xrec.append(AHU@AHU.conj().T)
    return xrec,sim
def show_layout_plots(layouts,farfield=False):
    """
    Shows position of transmitters and receivers, visibility dirstribution, and point spread function for every radar layout in the list layouts
    Viser radaroppsett med tilhørende funksjoner for hvert oppsett
    INNDATA:
    layouts - liste med navnet på radaroppsettene som ønskes.
    """
    for r,layout in enumerate(layouts):
        radar = Radar(layout=layout)
        radar.layoutplots(angle = 5,recons = 'MF',plot_virtrec=False,farfield=farfield,rader=len(layouts),radnå=r+1)
class Radar:
    """
    Inneholder sender- og mottakerposisjoner
    Contains transmitter and reciever posistions"""
    
    layouts = [
    # Éi antenne i kjerna sender, alle andre mottar
    {'tpos': eiscat3dtpos[:,5:6],
     'rpos': hstack([eiscat3dtpos[:,0:5],eiscat3dtpos[:,6:],eiscat3drpos]),
     'maks': 0,
     'name': 'singtrans'
     },
    # Éi antenne i kjerna sender, alle andre mottar, bare kjerna
    {'tpos': eiscat3dtpos[:,5:6],
     'rpos': hstack([eiscat3dtpos[:,0:5],eiscat3dtpos[:,6:]]),
     'maks': 0,
     'name': 'singtranscore'
     },
     # Tre antenner i kjerna sender, alle mottar
    {'tpos': eiscat3dtpos[:,(25,90,60)],
     'rpos': hstack([eiscat3dtpos,eiscat3drpos]),
     'maks': 0,
     'name': '3trans'
     },
    # Tre antenner i kjerna sender, alle i kjerna mottar
    {'rpos': eiscat3dtpos,
     'tpos': eiscat3dtpos[:,(25,90,60)],
     'maks': 0,
     'name': '3transcore'
     },
    # Mercedesstjerne, radaren i Skibotn
    {'tpos': null,
     'rpos': mersestjerne,
     'maks':0,
     'name':'MBS'
        },
    # Bokstaven T
    {'tpos': null,
     'rpos': TE,
     'maks': 0,
     'name': 'te'
     },
    # Firkant
    {'tpos':null,
     'rpos':boks,
     'maks':0,
     'name':'kvadrat'
     },
    # En T med pigger
    {'tpos':null,
     'rpos':Tto,
     'maks':0,
     'name':'T2'
     },
    # Test2
    {'tpos':array([[0,1,0]]).T,
     'rpos':array([[0,-1,0],[-1,0,0]]).T,
     'maks':0,
     'name':'test2'
     }
    
    ]
    # Number of radar layouts
    nlayouts = len(layouts)
    
    @staticmethod
    def getlayout(n):
        """
        Gir ut antenneopplegg basert på navn eller nummer
        Gives out antenna layout based on name or number
        
        INNDATA:
        n - oppleggnavn/-nummer
        UTDATA:
        tpos,rpos,maks,name - sender- og mottakerposisjoner, største oppløsning (som programmet klarer) og navn.
        """
        if isinstance(n,int):
            layout = Radar.layouts[n]
        elif isinstance(n,str):
            for layout in Radar.layouts:
                if layout['name'] == n:
                    break
        tpos = layout['tpos']
        rpos = layout['rpos']
        maks = layout['maks']
        name = layout['name']
        return tpos,rpos,maks,name
    
    def __init__(self,layout=-1,tpos=eiscat3dtpos,rpos=eiscat3drpos,theta=1,f=230e6,show=False):
        """
        Lager ny radar
        Creates new radar using the parameters given. The "layout"-parameter overrides tpos and rpos.
        
        INNDATA
        tpos - (Valgfri) Posisjon til senderne [m], 3xn - matrise med posisjonene til senderne. Standard: eiscat3dtpos (se lenger oppe)
        rpos - (Valgfri) Posisjon til mottakerne [m], 3xm - matrise med posisjonene til mottakerne. Standard: eiscat3drpos (se lenger oppe)
        theta- (Valgfri) Åpningsvinkel til radaren [grader], skalar. Standard = 1 grad
        f    - (Valgfri) Radarfrekvens, skalar. Standard = 230 MHz
        layout-(Valgfri) Radaroppsett, et tall mellom 0 og Radar.nlayouts-1. Overskriver tpos og rpos. 
        show - (Valgfri) Angir om radarposisjonene skal vises med én gang. Standard: False
        
        UTDATA
        Radar - radarobjekt med egenskapene til radaren
        
        
        """
        
        if layout != -1:
            tpos,rpos,junk,crap = Radar.getlayout(layout)

        else:
            # Størrelse til sendermatrisa
            R,K = shape(tpos)
            # Størrelse til mottakermatrisa
            Rm,Km = shape(rpos)
            
            if any((R != 3,Rm != 3)):
                ValueError('The positional matrices must have three rows (with coordinates!)')
        self._tpos = tpos
        self._rpos = rpos
        self.theta= theta
        self.f = f
        self.layout = str(layout)
        if show:
            figure()
            self.show()
        
    def virtrec(self,show=True,letter=None):
        """
        Regner ut plasseringa til virituelle mottakere for MIMO-radar
        Calculates position of virtual recievers in MIMO radar
        """
        
        Tx = self._tpos[0,]
        Ty = self._tpos[1,]
        Tz = self._tpos[2,]
        
        Rx = self._rpos[0,]
        Ry = self._rpos[1,]
        Rz = self._rpos[2,]
                
        dx = -diffcomb(Tx,Rx)
        dy = -diffcomb(Ty,Ry)
        dz = -diffcomb(Tz,Rz)
        
        d = vstack([dx,dy,dz])


        if show == True or show == 'only' or show == 'notnewplot':
            print("plotting...")
            if show != 'notnewplot':
                figure()
            scatter(dy,dx,c = 'b',s = 100,marker='x')
            xlabel('East [m]')
            ylabel('North [m]')
            if letter is None:
                title('Virtual receivers')
            else:
                title(letter+') Virtual receivers')
            
#            show_plots()
        
        if not show == 'only':
            return d
    def visibility(self,show=True,virtrec=None,new=False,letter=None):
        """
        Regner ut alle posisjonsforskjellene mellom senderne og mottakerne
        Calculates all position displacements between the transmitter and the receivers
          
        INNDATA:
        show - Angir om synligheta skal vises. Når show = 'only' utgis ikke synlighetsdataene
            - show = 'only' does not hand out the visibilities (only plots them)
        virtrec - oppgi ferdig utregna virituelle mottakere her. Standard: None (ikke oppgitt)
            - virutal receivers, nott needed, but the computation becomes faster if these are calculated already
        new - påtving nye beregninger (ikke se i filarkivet om det er blitt gjort før)
            - force new calculations (don't look for these in existing files)
        letter - Hvilken bokstav som skal stå i tittelen. Det er praktisk dersom flere figurer plottes i ett. Standard: None, dvs ingen bokstav
            - Which letter to use in the title (for figures shown in articles where this is obligatoric)
        """
        
        if self.layout == "-1":
            new=True
        
        if virtrec is None:
            virtrec = self.virtrec(show=bool(show))
        
        filnavn = fillagresti + 'sikt/' + str(self.layout) + '.hdf5'
            
            
        if is_hdf5(filnavn) and not new:
            print('Found visibilities. Loading...')
            with File(filnavn,'r') as f:
                vu = asarray(f['vu'])
                vi = asarray(f['vi'])
                vc = asarray(f['vc'])
                vf = asarray(f['vf'])
        else:
            vx = diffcomb(virtrec[0,])
            vy = diffcomb(virtrec[1,])
            del(virtrec)
            
            # Visibilities
            v = vx.round(decimals=1)+vy.round(decimals=1)*i
            
            # Counting
            vu1,vf1,vi1,vc1 = unique(v,return_counts=True,return_inverse=True,return_index=True)
            vu,vi,vc,vf = uniktol(vstack([vu1.real,vu1.imag]),ac=vc1,ai=vi1,ai2 = vf1,tol = 0.5)
            
            with File(filnavn,'w') as f:
#                print('Lagrer teorimatrise til neste gang...')
                f.create_dataset("vu", data=vu )
                f.create_dataset("vi", data=vi )
                f.create_dataset("vc", data=vc )
                f.create_dataset("vf", data=vf )
        # Plotting
        if show == True or show == 'only' or show == 'notnewplot':
            
            print("plotting...")
            if show != 'notnewplot':
                figure(figsize=(15,12))
#            scatter(vu.imag,vu.real,c=vc,cmap='jet')
            scatter(vu[1,],vu[0,],c=vc,cmap='jet')
            xlabel('East [m]')
            ylabel('North [m]')
            if letter is None:
                title('Visibility')
            else:
                title(letter + ') Visibility')
            
            cbar = colorbar()
            cbar.set_label('Number of measurements', rotation=270)
#            show_plots()
        
        if not show == 'only':
            return vu,vi,vc,vf
#    def longest_baseline(self,virtrec=None):
#        """
#        Regner ut lengste bakkeavstand til radaren dvs. avstanden mellom de fjerneste virtuelle mottakerne
#        """
#        if virtrec is None:
#            virtrec = self.virtrec(show=False)
#
#        vx = Radar.diffcomb(virtrec[0,])
#        vy = Radar.diffcomb(virtrec[1,])
#        
#        baselines = sqrt(vx*vx+vy*vy)
#        
#        lbl = baselines.max()
#        
#        return lbl
    
    def traveltimes(self,targetpos):
        """
        Regner ut reisetid for signal fra alle senderne S_1-S_n til alle mottakerne M_1-M_m gjennom plasmaboblene P_1-P_k
        Calculates travel time for signal from all transmitters to all recievers
        
        
        I svarmatrisa går plasmaklumpene langs x-aksen og Sender-mottaker-kombinasjonene langs y-aksen:
            S1M1P1   S1M1P2 ... S1M1Pk
            S1M2P1   S1M2P2 ... S1M2Pk
            .
            S1MmP1   S1MmP2 ... S1MmPk
            S2M1P1   S2M1P2 ... S2M1Pk
            .
            S2MmP1   S2MmP2 ... S2MmPk
            .
            SnMmP1   SnMmP2 ... SnMmPk

            
            Antar konstant reisefart lik lyshastigheta.
        
        INNDATA
        plasmapos - posisjon til radarmålene [m], 3xNp - matrise
        
        UTDATA
        T - reisetid til alle sender-plasma-mottaker-kombinasjonene [s], (Nm*Ns) x Np-matrise. 
        
        """    
        # Størrelse til sendermatrisa
        Rt,Kt = shape(self._tpos)
        # Størrelse til mottakermatrisa
        Rm,Km = shape(self._rpos)
        # Størrelse til plasmamatrisa
        Rp,Kp = shape(targetpos)
        if Rp != 3:
            ValueError('Plasmaposisjonsmatrisa må ha tre rader (med koordinater)!')
            ValueError('The plasma position matrix must have three rows (with coordinates)!')
        
       
        trans = repeat(self._tpos,Km*Kp,axis=1)
#        print(trans)
        rec   = repmat(repeat(self._rpos,Kp,axis=1),1,Kt)
#        print(rec)
        plasma= repmat(targetpos,1,Kt*Km)
#        print(plasma)
       
        # Reiselengder
        L = norm(plasma-trans,axis=0) + norm(rec-plasma,axis=0)

        # Reisetid
        T = (L/c).reshape(Kt*Km,Kp)
        
        return T
    
    def theorymatrix(self,plasma,inv_comp=False,new=False):
        """
        Regner ut verdiene i teorimatrisa A
        Calculates values in theory matrix A
        
        plasma - scatter points (a plasma)
        inv_comp - get out the inverse of the compression (so that equal measurements can be extended again) - Will not work for nearfield
        new - Force new calculation of theory matrix
        
        """
        farfield = plasma.farfield
        # (Antatt) plasmaoppløsning
        pr = int(sqrt(plasma.val.size))
        prt = '_ps-'+str(pr)+'x'+str(pr)
        
        if self.layout == "-1":
            new = True
        
        filnavn = fillagresti + 'teorimat/L-' + str(self.layout) + '_' + 'rad-' + str(round(plasma.radius)) + '_hkm-'+str(round(plasma.height/1e3)) + prt
        if not farfield:
            filnavn += '_nf.hdf5'
        else:
            filnavn += '_ff.hdf5'
            
            
        if is_hdf5(filnavn) and not new:
            print('Found theory matrix. Loading...')
            with File(filnavn,'r') as f:
                A = asarray(f['A'])
                if farfield:
                    vi = asarray(f['vi'])
                    vf = asarray(f['vf'])

        else:
#            print('Har ikke teorimatrise. Beregner den...')
            if not farfield:
                
                # Størrelse til sendermatrisa
                Rt,Kt = shape(self._tpos)
                # Størrelse til mottakermatrisa
                Rm,Km = shape(self._rpos)
                
                T = float32(self.traveltimes(plasma.pos))
                # Størrelse til reisetidsmatrisa
                RT,KT = T.shape
                
                try:
                    A = zeros([RT*RT,KT],dtype='complex64')
                    for rad in range(len(A)):
                        T2mT3 = T[rad // RT,] - T[rad % RT,]
                        A[rad,] = complex64(exp(i*2*pi*self.f*T2mT3))
                        del T2mT3
                    del T
                except MemoryError as e:
                    print('MemoryError by creating a',RT*RT,'x',KT,'matrix')
                    raise e
                
                   
            else:
                angx = arctan2(plasma.pos[2,],plasma.pos[1,])
                angy = arctan2(plasma.pos[2,],plasma.pos[0,])
                
                
                tetax,tetay = cos(angx),cos(angy)
                tetaz = tetax**2+tetay**2
                
                k = float32(2*pi*self.f/c * vstack([tetax.reshape(1,-1,1),tetay.reshape(1,-1,1),tetaz.reshape(1,-1,1)]))
    
                vu,vi,vc,vf = self.visibility(False)
    
                        
                v = float32(vstack([vu,zeros(len(vc))]).reshape(3,1,-1))
        
                A = exp(i*asum(k*v,axis=0)).T
                
            with File(filnavn,'w') as f:
#                print('Lagrer teorimatrise til neste gang...')
                f.create_dataset("A", data=A )
                if farfield:
                    f.create_dataset("vi", data=vi )
                    f.create_dataset("vf", data=vf )

        if inv_comp:
            return A,vi,vf
        else:
            return A
    def show(self,scat=False,at_once=True,plot_numbers=False,letter=None):
        """
        Viser posisjonen til radarantennene. Figuren projiseres på planet der z=0
        Shows the position of the radar antennas. The figure is projected on the plane where z=0
        
        INNDATA
        scat - angir om plottet skal være et spredningsplott. Det kan være at plottinga går saktere da. (standard:False)
        at_once - Skal plottet komme opp med én gang? Standard: True
        plot_numbers - Om nummeret til antenna skal plottes sammen med antenna (Standard:False)
        letter - Hvilken bokstav som skal stå i tittelen. Det er praktisk dersom flere figurer plottes i ett.. Standard: None, dvs ingen bokstav
        """
    
        if scat:
            scatter(self._rpos[1,],self._rpos[0,],c = 'b',s = 100,marker='x',label = "Receivers")
            scatter(self._tpos[1,],self._tpos[0,],c = 'r',s = 100,marker='+', label = "Transmitters")
        else:
            plot(self._rpos[1,],self._rpos[0,],'x',color = 'b',label = "Receivers")
            plot(self._tpos[1,],self._tpos[0,],'+',color = 'r', label = "Transmitters")
        legend()
        ylabel('North [m]')
        xlabel('East [m]')
        if not letter is None:
            title(letter+')')
        
        if plot_numbers:
            ax = gca()
            for i1,r in enumerate(self._rpos.T):
                ax.text(r[1],r[0],str(i1))
        
        
        if at_once:
            show_plots()
        
    def create_measurement(self,plasma,inv_comp=False):
        """
        Simulerer ei radarmåling med plasmaet som beskrevet i plasma-objektet
        Simulates a radar measurement with the target as described by the plasma object
        
        plasma - scatter points (a plasma)
        inv_comp ------- turned off------ get out the inverse of the compression (so that equal measurements can be extended again)
        
        """
        x = plasma.val.T
        
#        if plasma.farfield != inv_comp:
#            raise ValueError('inv_comp must må oppgi om plasmaet er i fjernfeltet! ')
        

                    
        if plasma.farfield:
#            raise ValueError('Plasmaet er i fjernfeltet, men dette støttes ikke her!')
            
            A,vi,vf = self.theorymatrix(plasma,inv_comp=True)
            # Måling
            m = (A@x)
#            # Normalizing
#            # Number of effective receivers
#            effrec = int(sqrt(len(vi)))
#            
#            # Normalizing crosscorrelations
#            m_mat = m[vi].reshape(effrec,-1)
##            mc = m[vi]/(trace(m_mat)/m_mat.shape[0])
#            mn = m/(trace(m_mat)/m_mat.shape[0])
            
#            return mn,vi
            return m,vi
        else:
            try:
                A = self.theorymatrix(plasma,inv_comp=False)
                # Måling
                m = (A@x)
            except MemoryError:
                plasmapart = Plasma(plasma.height,1,coord='xy',cut=False,farfield = plasma.farfield)
                
                #preallokering
                a = self.theorymatrix(plasmapart,inv_comp=False)
        
                m = zeros(a.shape,dtype='complex128')
                
                # Legger målinga av hvert punkt til den oppsummerte radarmålinga
                for i1,point in enumerate(plasma.pos.T):
                    plasmapart.pos = point.reshape(3,-1)
                    a = self.theorymatrix(plasmapart,inv_comp=False,new=True)
                    
                    m += a*x[i1]
                    if i1%100 == 0:
                        print(i1)
            
#            # Normaliserer målingene
#            effrec = int(sqrt(len(m)))
#            m_mat = m.reshape(effrec,-1)
#            mn = m/(trace(m_mat)/m_mat.shape[0])
        
#            return mn
            return m
    def get_meas(self,im='nordlys',farfield=False,sigma=0.05,new=False):
        """
        Lager/laster inn et målesett.
        Creates/loads a measurement set. This code does currently not represent the final state since these files currently are unavailable. 
        
        INNDATA:
        layout - Hvilket radaroppsett som brukes.
            - The radar layout
        im - Hva som er tatt som originalbilde (Standard: Nordlys (97x97 piksler))
            - brightness distribution oif target
        farfield - Angir om målingene skal gjøres i fjernfeltet til radaren (Standard: False)
            - For measurements in the farfield of the radar
        sigma - (Relativt) Standardsavvik til støy. (Standard: 0,05)
            - relative standard deviation of noise
        new - Påtving ny måling. Standard: False, med mindre radaroppsettet er "-1"
            - Force new measurement calculations. This is also enabled by layout=-1
        
        UTDATA
        m - målesettet, vektor
        x - originalbildet, vektor
        
        """
        
        if self.layout == "-1":
            new = True
        

        
        # Filnavn til denne målinga
        if farfield:
            filnavn = fillagresti + 'meas_'+self.layout+'_'+im+'_far'+'.hdf5'
        else:
            filnavn = fillagresti + 'meas_'+self.layout+'_'+im+'_near'+'.hdf5'
        
        # Ser etter som vi har målinger
        if is_hdf5(filnavn) and not new:
            print('Found measurements. Loading...')
            with File(filnavn, "r") as f:
                m = asarray(f['meas'])
                x = asarray(f['true'])
        #                At = asarray(f['thrm'])
                if farfield:
                    vi = asarray(f['vi'])
                
#            target = x.reshape(int(sqrt(x.size)),-1)
            
        else: # Lager nye målinger
            print('Did not find measurements. Creates new ones.')
            # Den ukjente
            target = get_target(im,True)
            
            # Lager plasma
            plasmat = Plasma(height=100e3,opening=1,coord='xy',res=target.shape,farfield=farfield,val=target,show=False,cut=False)
            
            x = plasmat.val.T
            
            if farfield:
                m,vi = self.create_measurement(plasmat,farfield)
            else:
                m = self.create_measurement(plasmat,farfield)
            
            # Lagrer
            with File(filnavn, "w") as f:
                f.create_dataset("meas", data=m )
                f.create_dataset("true", data=x )
        #                f.create_dataset("thrm", data=At )
                if farfield:
                    f.create_dataset("vi", data=vi)
        # Normaliserer målingene
#        m /= get_target(im,True).size
        m /= get_target(im,True).size########################################################################################
#        m = m/sqrt(m[0])
        # Legger til støy, merk at flere like målinger får samme støy (gjelder fjernfeltsmålinger)
#        m = m.real*( 1+sigma*standard_normal([m.shape[0],1]) ) + m.imag*( 1+sigma*standard_normal([m.shape[0],1]) )*i
        m += m.real.max()*sigma*standard_normal([m.shape[0],1]) + m.imag.max()*sigma*standard_normal([m.shape[0],1])*i

        if farfield:
            mn = normalize_measurements(m,vi)
            return mn,x,vi
        else:
            mn = normalize_measurements(m)
            return mn,x   
    def recover(self,A,m,vi=None,method='MF',methodparam=None,plasma=None,new=False):
        """
        Rekonstruerer et bilde
        resonstructs an image
        
        INNDATA:
        A - Teorimatrise
            - theory matrix
        m - Målinger
            - measuremnts
        vi - invers komprimeringsvektor. Når A og m er komprimerte (fordi mange 
            grunnlinjer er like), må de dekomprimeres for å normalisere målingene. 
            Dersom vi ikke er oppgitt, antas det at målingene er normaliserte!
            - for compressed A and m (is normally not the case)
        method - Hvilken rekonstrusksjonsmetode som skal brukes (Standard: Tilpassa filter, 'MF')
            - Method for reconstructing the image (Currently "MF", "capon", "TSVD", and "LS" are available)
        methodparam - Parametre for deon brukte metoden (Kan utelates for enkelte metoder)
            - Paramaters for the chosen reconstruction method (currently only used for TSVD)
        
        UTDATA:
        xh - Rekonstruert bilde
        """
        
        if isinstance(plasma,Plasma) and not method in ['MF','capon']:
            # Åpne lagra fil
            
            # (Antatt) plasmaoppløsning
            pr = int(sqrt(plasma.val.size))
            prt = '_ps-'+str(pr)+'x'+str(pr)
        
            filnavn = fillagresti + 'tminv/L-' + str(self.layout) + '_' + 'rad-' + str(round(plasma.radius)) + '_hkm-'+str(round(plasma.height/1e3)) + prt
            if not plasma.farfield:
                filnavn += '_nf-'
            else:
                filnavn += '_ff-'
            filnavn += method + str(methodparam)+'.hdf5'
            if is_hdf5(filnavn) and not new:
                lastinn = True
            else:
                lastinn=False
        else:
            lastinn = False
            
            
        if lastinn:
            #print('Hadde invers teorimatrise. Laster inn...')
            print('Found inverse theory matrix. Loading...')
            with File(filnavn, "r") as f:
                Ainv = asarray(f['ainv'])
            xh = Ainv@m
        else:
            if method == 'MF':
                xh = MF(A,m)
            elif method == 'capon':
                xh = capon(A,m,vi,normalize=False)
            elif method == 'TSVD':
                Ainv,xh = TSVD(A,m,methodparam)
            elif method in ['MKM','LS']:
                Ainv,xh = MKM(A,m)
            else:
#                raise ValueError('Kunne ikke bruke ønska metode('+method+') da den ikke er implementert, evt. sjekk stavemåten!')
                raise ValueError("Couldn't use the desired method ("+method+") because it is not implemented. Please check the spelling!")

        
        if isinstance(plasma,Plasma) and not lastinn and not method in ['MF','capon']:
            # Lagrer invers matrise til neste gang
            with File(filnavn, "w") as f:
                f.create_dataset("ainv", data=Ainv )
        
        return xh
    def comprec(self,x,plasma,A,m,methods,methodparams,vi=None,show=True,normcomp=False):
        """
        Regner ut intensiteten på valgte måter og plotter dem.
        Calculates the image by chosen methods and plots them
        
        x - Opprinnelig bilde
            - original image
        plasma - Plasmaet slik vi trur det er og vil finne ut elektrontettheta til.
            - target plasma object
        A - teorimatrise til forsøket. Størrelsen gjenspeiler oppløsninga til resultatet
            - theory matrix. The imaging resolution is defined by this theory matrix
        m - målevektor
            - measurement vector
        methods - liste med metoder som brukes for å regne ut intensiteten
            - list with methods used to find the image
        methodparams - liste med parametre til hver enkelt metode
            - Corresponding parameters
        
        """
        
        
        
        oppl = int(sqrt(x.shape[0]))
        # "Denormaliserer m"
        m *= plasma.val.size########################################################################################
        
        plasmat = Plasma(plasma.height,plasma.opening,coord='xy',res=(oppl,oppl),val=None,farfield=plasma.farfield,show=False,cut=False)
        
        if show:
            figure(figsize=(14,10))
            subplot(2,2,1)
            plasmat.showestim(sol=x,t='Original image',cmap='jet')
        
        xrec = []
        sim = []
        for nm,method in enumerate(methods):
            xh = self.recover(A,m,vi,method,methodparams[nm],plasma)
            xrec.append(xh)
            ulikhet = compimag(x.reshape(oppl,oppl),xh.real.reshape(int(sqrt(xh.shape[0])),-1),norm=normcomp)
            sim.append(ulikhet)
            if nm < 3 and show:
                subplot(2,2,2+nm)
                plasma.showestim(xh.real,method,'jet')
            
        return xrec,sim
    
    def psf(self,angle = None,show = True,farfield=False,rekons='MF',forcenew=False,letter=None):
        """
        Plotter punktspredningsfunksjonen
        Plots the point spread function
        
        INNDATA
        angle - Hvor stort område (i grader i hver retning) punktspredningsfunksjonen skal dekke (Standard: radarstrålen)
            - What area the PSF should cover. Default: HPBW
        show - Om punktspredningsfunksjonen skal plottes (True (ja),False (nei),'only' (ikke gi ut resultatet, bare plott) eller 
                'notnewplot' (ikke lag ny figur)), Standard: True
                - Plot PSF (True/False), 'only' means plot, but don't return the psf, or 'notnewplot' for plotting into existing figure.
        farfield - Foregår målingene i fjernfeltet? Standard: True
            - use farfield for PSF
        rekons - Brukes med omhu - Hvilken rekonstruksjonsteknikk som skal brukes. Standard: 'MF' (tilpassa filter). 
                Andre også tilgjengelig, men kan føre til minneproblemer m.m.
                - Use with care - What reconstruction method to use. Standard: matched filter
        letter - Hvilken bokstav som skal stå i tittelen. Det er praktisk dersom flere figurer plottes i ett.. Standard: None, dvs ingen bokstav
                - letter for the title. Practical for journal articles.
        """
        if angle is None:
            angle = self.theta/2
        
        target = get_target('psf',True)
        plasma = Plasma(100e3,angle*2,'xy',target.shape,target,False,False,False)
        print(plasma.radius,angle*2,arctan2(plasma.radius,plasma.height))
        if farfield:
            plasma.farfield = True
            
            m,x,vi = self.get_meas(im='psf',farfield=True,sigma=0,new=forcenew)
            A,vi,vf = self.theorymatrix(plasma,inv_comp=True)
            psf = self.recover(A,m,vi,method='MF')
        else:
            m,x = self.get_meas(im='psf',farfield=False,sigma=0,new=forcenew)
            A = self.theorymatrix(plasma,inv_comp=False)
            psf = self.recover(A,m,method='MF')
            
        psf_abs = sqrt(psf.real**2 + psf.imag**2)
        
        if show != False:
            if show != 'notnewplot':
                figure(figsize=(10,8))
            # Senterpiksel
            psf_cent = psf_abs[tuple((array(psf_abs.shape)/2).astype('int64'))]
            
            plasma.showestim(10*log10(psf_abs/psf_cent),angles=True,cmap='jet',cb=False,cl=(-40,0))
            clim(-40,0)
            colorbar(label='Power (dB)')
            if letter is None:
                title('Point spread function')
            else:
                title(letter+') Point spread function')
            
            
        if show != 'only':
            return psf
        
    def layoutplots(self,angle=None,recons='MF',plot_virtrec=True,newmeas=False,farfield=False,rader=1,radnå=1):
        """
        Viser plott for at hvert oppsett kan vurderes
        Shows plots for one layout
        
        INNDATA
        angle - Hvor stort område (i grader i hver retning) punktspredningsfunksjonen skal dekke (Standard: radarstrålen)
            - field of view of the PSF, for the plotting, not the radar itself
        plot_virtrec - overrides plotting into existing figure and plots four plots for every layout, including virtual receivers
        farfield - Foregår målingene i fjernfeltet? Standard: True
        newmeas - Om det skal bli tvungen nymåling
            - force new measurements (don't load them from disc)
        rader - Hvor mange oppsett som plottes i hver figur. Er ikke betydningsfull hvis plot_virtrec=True (hvis virtuelle mottakere vises). Standard = 1.
            - if these plots are a part of a larger plot, this variable tells the row number of this plot (uses subplot - one-based indexing!!!)
        radnå - Hvilken rad som plottes nå. Er ikke betydningsfull hvis plot_virtrec=True (hvis virtuelle mottakere vises). Standard = 1.
            - What row  number is plotted currently
        """
        alfabetet = [chr(i) for i in range(ord('a'),ord('{'))]+['æ','ø','å']
        print('')
        #print('Oppsettsplott for oppsett',self.layout,':')
        print('Layout plot for layout',self.layout,':')

        if angle is None:
            
            angle = self.theta/2
        
        # Klargjør for plotting/ Lager ramme
        if plot_virtrec:
            figure(figsize = (14,10))
            subplot(2,2,1)
            plottnr=1
        else:
            # Ny figur kun for plotting av første rad
            if radnå==1:
                figure(figsize = (16,4.5*rader))
                subplots_adjust(hspace=0.3,wspace=0.3)
            plottnr = (radnå-1)*3 + 1
            subplot(rader,3,plottnr)
        # Plotter radaren
        self.show(at_once=False,letter=alfabetet[plottnr-1])
        
        # Plotter virituelle mottakere
        #print('Setter ut virituelle mottakere...')
        print('Placing virtual receivers...')

        if plot_virtrec:
            
            subplot(2,2,2)
            d = self.virtrec(show='notnewplot',letter=alfabetet[2-1])
            
            subplot(2,2,3)
            plottnr = 3
        else:
            d = self.virtrec(show=False)
            plottnr = (radnå-1)*3 + 2
            subplot(rader,3,plottnr)
        # Plotter sikta
        #print('Regner ut sikta')
        print('Calculating visibility')
        vu,vi,vc,vf = self.visibility(show = 'notnewplot',virtrec = d,letter=alfabetet[plottnr-1])
        if plot_virtrec:
            plottnr=4
            subplot(2,2,4)
        else:
            plottnr = (radnå-1)*3 + 3
            subplot(rader,3,plottnr)
        # Plotter punktspredningsfunksjonen
        print('Calculating point spread function')
        #print('Regner ut punktspredningsfunksjon')

        psf = self.psf(angle=angle,show = 'notnewplot',farfield=farfield,rekons = recons,forcenew=newmeas,letter=alfabetet[plottnr-1])
        
        del psf
    def basiscov(self,plasma,method='MKM',methodparam=None,sd=0.05,show_cov=False,show_sd=True):
        """
        Regner ut kovariansen til målingene
        Calculates (and plots) covariance of measurements
        
        INNDATA
        plasma - plasmaobjekt
        method - Hvilken rekonstrusksjonsmetode som skal brukes (Standard: Tilpassa filter, 'MF')
        methodparam - Parametre for deon brukte metoden (Kan utelates for enkelte metoder)
        sd - (Relativt) Standardavvik til støy. (Standard: 0,05)
        show_cov - Sier om kovariansene skal plottes
        show_sd - Sier om standardavviket skal plottes
        """
        if plasma.farfield:
            print('OBS: The matrix AH*A might not be invertible!')
            #print('OBS: Det kan være at matrisa AH*A ikke er inverterbar!')

        if sd == 0:
            print('OBS: Noise-free measurements are assumed. ')
            #print('OBS: Det er antatt støyfrie målinger. Kan derfor ikke vise ')
            show_sd=False
        A = self.theorymatrix(plasma,inv_comp=False)

        if method == 'TSVD':
            pass
        else:# method == 'MKM':
            try:
                cov = inv(A.conj().T@A)
            except MemoryError:
                print('Multiplies matrix row for row.')                                
                #print('Tar matrisemultiplikasjonen rad for rad.')

                M = matmulfor(A.conj().T,A)
                
                cov = pinv(M)
            except LinAlgError:
                print('Matrix is singular. Covariance is nan')
                #print('Matrisa er singulær. Kovariansmatrisa blir nan')
                cov = ones((A.shape[1],A.shape[1]))*nan

        S = cov*sd*sd
        if show_cov:
            figure(figsize=(10,8))
            pcolormesh(S.real)
            colorbar()
        if show_sd:
            # Kun varianser
            s = S.real.diagonal().reshape(-1,1)
            figure()
            # Plotter standardavviket
            plasma.showestim(sqrt(s),'Standard deviation for least squares solution','jet')
        return S

def rec_plots(layouts,target,sigma=0.05,h=100e3,o=1,N=[18,20],methods=['MKM'],methodparams=[None],farfield=False,showres=True,showsds=True,showcovplot=True,normcomp=False):
    """
    Går gjennom ulike oppsett og antatte bildestørrelser og måler usikkerheta til resultatet.
    Looprs over different layouts and imaging resolutions and measures uncertainty of the result
    
    methods - må være liste eller 'svddiag'
            - must be a list or 'svddiag' (if the latter, svddiag is called for the imaging, else Radar.comprec is called)
    """
    with Timer('Preallokering'):
        if isinstance(N,int):
            N = [N,N]
            showcovplot = False
        
        # Midlere standardavvik    
        sm = zeros((len(layouts),N[1]-N[0]+1))
        # Midlere standardavvik til TSVD
        sSVD = zeros(sm.shape)
        sT21 = zeros(sm.shape)
        sT210= zeros(sm.shape)
        # Residualkvadratsum
        sse = zeros(sm.shape)
        # Estimert standardavvik på målingene
        shatt = zeros(sm.shape)
        # Midlere kvadratavvik fra "sant" bilde
        if isinstance(methods,list):
            mse = zeros((len(layouts),N[1]-N[0]+1,len(methods)))
        else:
            mse = zeros((len(layouts),N[1]-N[0]+1,4))

        Nv = range(N[0],N[1]+1)
    for n in Nv:
        with Timer(('round'+str(n))):
            res = (n,n)
            plasma = Plasma(height=h,opening=o,coord='xy',res=res,val=None,show=False,cut=False,farfield=farfield)
            for nl,layout in enumerate(layouts):
                radar = Radar(layout=layout)
                m,x = radar.get_meas(target,farfield=farfield,sigma=sigma)
                
#                m *= (2*h*tan(o*pi/360)/n)**2
#                m *= n*n

                # Bruk målingene
                
                if farfield:
                    A,vi,vf = radar.theorymatrix(plasma,inv_comp=True)
    #                radar.comprec(x,plasma,A,m,methods,methodparams,vi) 
                else:
                    A = radar.theorymatrix(plasma,inv_comp=False)
                    vi = None
                
                if showres == 'save' or showres:
                    shownow=True
                else:
                    shownow=False
                    
                
                if methods == 'svddiag':
                    xrec,sim = svddiag(x,plasma,A,m,rcond = methodparams,show=shownow,normcomp=normcomp,layout=layout)
    #                covTSVD = xrec[1]*sigma*sigma
                    sSVD[nl,n-N[0]] =sigma*sigma*xrec[1].diagonal().real.mean()
                    sT21[nl,n-N[0]] =sigma*sigma*xrec[2].diagonal().real.mean()
                    sT210[nl,n-N[0]]=sigma*sigma*xrec[3].diagonal().real.mean()
                else:        
                    xrec,sim = radar.comprec(x,plasma,A,m,methods,methodparams,vi,show=shownow,normcomp=normcomp) 
                if showres=='save':
                    navn = fillagresti+'plots/'+layout+'_'+str(res)+'_result.png'
                    savefig(navn)
                    clf()
                    close('all')
                
                if showcovplot or showcovplot=='save':
                    # Beregner residualkvadratsummen
                    if isinstance(methods,list):
                        idxMKM = methods.index('MKM')
                    else:
                        idxMKM = 0
                    eps = m-A@xrec[idxMKM] # Residualer
                    SSE = (eps.conj().T @ eps).real[0,0] # Residualkvadratsum
                    SIGMA = sqrt(SSE / (m.size-xrec[idxMKM].size) ) # estimert standardavvik
                    print('m.size-xrec[idxMKM].size',m.size-xrec[idxMKM].size,'SIGMA',SIGMA)
                    if SIGMA < 0:
                        print('Sigma is ',SIGMA,'and becomes therefore nan')
                        #print('Sigma er ',SIGMA,'og settes derfor til nan')

                        SIGMA = nan
    #                SIGMA[SIGMA <0] = nan
                    # Beregn og plott kovarians / standardavvik
                    S = radar.basiscov(plasma,sd=SIGMA,show_cov=False,show_sd=showsds)
                    # Standardavvik
    #                s = sqrt(S.real.diagonal().reshape(res))
    #                smean = s.mean()
                    smean = sqrt(S.real.diagonal().mean())
                    if not smean > 0:
                        #txst = ('Her har det skjedd noe skikkelig galt. (Den positive) kvadratrota av et reelt tall,'+str(S.real.diagonal().mean())+', ble negativ,'+str(smean), '!')
                        txst = ('Here something really wrong has happened. (The positive) square root of a real number,'+str(S.real.diagonal().mean())+', became negative,'+str(smean), '!')
    #                    print('Varianser til MKM-estimater',S.real.diagonal().reshape(res))
                        print(txst)
    #                    raise RuntimeError(txst)
                    # Midlere standardavvik
                    sm[nl,n-N[0]] = smean
                    shatt[nl,n-N[0]] = SIGMA
                    sse[nl,n-N[0]] = SSE
                    mse[nl,n-N[0],:] = array(sim)
    with Timer('Plotting'):
        if showcovplot or showcovplot=='save':

            for nl,layout in enumerate(layouts):
                figure(figsize = (15,12))
                subplot(2,2,1)
                if isinstance(methods,list):
                    plot(array(Nv),sm[nl,:],label=layout)
                else:
#                    print(Nv,sm[nl,:])
                    try:
                        loglog(array(Nv),sm[nl,:]*sigma**2/shatt[nl,:]**2,label=layout+' LS')
                    except:
                        #print('Kunne ikke plotte variansen til MKM-estimatet!')
                        print("Couldn't plot variance of LS-estimate!")
                        print(array(Nv),sm[nl,:])
#                        loglog(array(Nv),ones(Nv[-1]-Nv[0]+1),label=layout+' LS')
                    try:
                        loglog(array(Nv),sSVD[nl,:] ,label=layout+' TSVD, rcond='+str(methodparams))
                    except:
                        #print('Kunne ikke plotte variansen til TSVD-estimatet!')
                        print("Couldn't plot variance of TSVD-estimate!")
#                        loglog(array(Nv),ones(Nv[-1]-Nv[0]+1),label=layout+' TSVD')
                    try:
                        loglog(array(Nv),sT21[nl,:] ,label=layout+' T2, alfa='+str(trp1))
                    except:
                        #print('Kunne ikke plotte variansen til T2_1-estimatet!')
                        print("Couldn't plot variance of T2-1-estimate!")
#                        loglog(array(Nv),ones(Nv[-1]-Nv[0]+1),label=layout+' T2, alfa=1')
                    try:
                        loglog(array(Nv),sT210[nl,:],label=layout+' T2, alfa='+str(trp2))
                    except:
                        #print('Kunne ikke plotte variansen til T2_1-estimatet!')
                        print("Couldn't plot variance of T2-1-estimate!")
#                        loglog(array(Nv),ones(Nv[-1]-Nv[0]+1),label=layout+' T2, alfa=1')
                title('Mean standard deviation s²(A''A)⁻¹')
                ylabel('Pixelintensity')
                xlabel('Assumed resolution [px]')
                legend()
                subplot(2,2,2)
#            for nl,layout in enumerate(layouts):
                plot(array(Nv),shatt[nl,:],label=layout)
                title('SD (measurement), estimated')
                ylabel('Power [W]')
                xlabel('Assumed resolution [px]')
                legend()
                subplot(2,2,3)
#            for nl,layout in enumerate(layouts):
                plot(array(Nv),sse[nl,:],label=layout)
                title('SSE')
                ylabel('Power [W]')
                xlabel('Assumed resolution [px]')
                legend()    
                subplot(2,2,4)    
#            for nl,layout in enumerate(layouts):
                if isinstance(methods,list):
                    for nm,method in enumerate(methods):
                        semilogy(array(Nv),mse[nl,:,nm],label=(layout+' '+method))
                elif methods == 'svddiag':
                    semilogy(array(Nv),mse[nl,:,0],label=(layout+' '+'MKM'))
                    semilogy(array(Nv),mse[nl,:,1],label=(layout+' '+'TSVD,rcond='+str(methodparams)))
                    semilogy(array(Nv),mse[nl,:,2],label=(layout+' '+'Tikhonov alfa='+str(trp1)))
                    semilogy(array(Nv),mse[nl,:,3],label=(layout+' '+'Tikhonov alfa='+str(trp2)))
                else:
                    semilogy(array(Nv),mse[nl,:,:],label=layout)
                title('Mean sq. error (diff. from orig., scal. image for comp.)')
                ylabel('Power² [W²]')
                xlabel('Assumed resolution [px]')
                legend()
                if showcovplot == 'save':
                    navn = fillagresti+'plots/'+layout+'_errors.png'
                    savefig(navn)
                    clf()
                    close('all')
                else:
                    show_plots()
                    
            
if __name__ == '__main__':
    layouts = ['singtranscore','singtrans','3transcore','3trans']
#    layouts = ['singtranscore','singtrans']

    with Timer('Oppsett'):
        # This will plot the layout plots in Figure 7
        show_layout_plots(layouts)
        show_plots()

        
        
        
        
        


        
        
        
        
        

