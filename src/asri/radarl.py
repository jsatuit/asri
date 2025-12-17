#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Radarligning o.l.
Radar equation and similar
"""

from asri.konstanter import kB,c,re,amu
from numpy import pi,sin,linspace,array,sqrt,ceil
from asri.plasma_funcs import debye,vth
from math import floor
import matplotlib.pyplot as plt

#Beregner spredningstverrsnitt

def sxv(f,Ne=3e11,Te=2000,Ti=1000,ki=90,r=100e3,dr=1e3,theta=1,p=False):
    """ 
    Beregner spredningstverrsnittet til et volum med elektroner
    Volumet befinner seg mellom avstand r og r+dr i strålen med bredde \theta. 
    Hvis theta=0 utgis spredningstverrsnittet til ett elektron.
    
    Calculates scattering cross-section of a volume with electrons
    The volume starts at range r and ends at r+dr in the beam with width \theta.
    If theta = 0, the function gives the scattering cross-section of one electron.
    
    
    INNDATA
    Obligatorisk:
    f - frekvens til innkommende signal [Hz], kan være skalar eller np.array
    
    Valgfri (skal være skalarer dersom ikke anna oppgitt):
    Plasma og spredning:    
    Ne - elektrontetthet [m⁻³].             Standard:3*10¹¹ m⁻³
    Te - elektrontemperatur [K].            Standard:2000K
    Ti - ionetemperatur [K].                Standard:1000K
    ki - spredningsvinkel [grader].         Standard:90 grader    
    Radar:
    r  - avstand til spredningsvolumet [m]. Standard:100 km
    dr - Tykkelse av spredningsvolumet [m]. Standard:1 km
    theta - full strålebredde [grader].          Standard:1 grad
    
    p   - Angir om verdier underveis (spredningsvolum) skal skrives ut. Standard: False
    UTDATA
    s0 - spredningstverrsnitt til ett elektron [m²], samme størrelse som w
    
    EKSEMPEL:
    # Beregner spredningstverrsnitt for ett elektron
    >>> sxv(f=c/1.4,Ne=3e11,Te=2000,Ti=1000,ki=90,theta = 0)
    3.340558588972082e-29
    
    #Beregner spredningstverrsnitt for et volum med elektroner som er 100 km langt unna med tykkelse 1 km
    >>> sxv(f=c/1.4,Ne=3e11,Te=2000,Ti=1000,ki=90,r=100e3,dr=1e3,theta=1)
    2.421640362462482e-08



    """

    #Spredningsvolum
    if theta == 0:
        V = 1/Ne
    else:
        #Eksakt: 
        #V = pi*dr*sin(theta*pi/180)**2/3 * (3*r**2 + 3*r*dr + dr**2)
        V = pi*dr*sin(theta*pi/360)**2/3 * (3*r**2 + 3*r*dr + dr**2)
        
        #Tilnærma :
        #V = pi*dr*sin(theta*pi/180)**2 * r**2
    

    #Debyelengde
    D = debye(Ne,Te)

    #Mulige bølgelengder til radaren
    bl = c/f
        
    alfa = 4*pi*D/bl
    
    #Spredningstverrsnitt:
    #Thomson
    se = 4*pi*(re*sin(ki*pi/180))**2
    
    #forsterkning?
    F = 1.0 - 1/(1+alfa**2) + 1/( (1+alfa**2) * (1+alfa**2+Te/Ti) )
    
    #Spredningstverrsnitt for alle elektronene
    s0 = V*Ne*F*se
    
    if p:
#        print("Thomson-spredningstverrsnitt:",se,"m²")
        if D.size <= 10:
            print("Debyelengde:",D,"m")
            print("Spredningstverrsnitt for ett elektron:",F*se,"m²")
        else:
            print("Debyelengde:",D[0:10],"(...) m")
            print("Spredningstverrsnitt for ett elektron:",F[0:10]*se,"(...) m²")
        if V.size <= 10:
            print("Spredningsvolum: ",V/1e9,"km³")
        else:
            print("Spredningsvolum: ",V/1e9,"(...) km³")
        

    
    
    return s0

   
def signalpower(Ne,Te,Ti,ki=90,r=100e3,dr=1e3,f=c/1.4,theta=1,Gt=10**4.3,Gr=10**2.2,Pt=5e6,p=False):
    """
    Beregner signalstyrken til mottatt signal
    Calculates signal power of recieved signal
    
    INNDATA
    Ionosfære
    Ne - elektrontetthet [m⁻³].
    Te - elektrontemperatur [K].
    Ti - ionetemperatur [K].
    ki - spredningsvinkel [grader].         Standard:90 grader
    r  - avstand til spredningsvolumet [m]. Standard:100 km
    dr - Tykkelse av spredningsvolumet [m]. Standard:1 km
    Radar
    f  - Radarsendefrekvens [Hz].           Standard:21437470 Hz, tilsvarer 1,4 m
    theta- Strålebredde [grader].           Standard:1 grad
    Gt - Forsterkning til sender [].        Standard:10^4,3
    Gr - Forsterkning til mottaker [].      Standard:10^2,2
    Pt - Utsendt effekt [W].                Standard:5*10⁶ W
    p   - Angir om verdier underveis (spredningstverrsnitt)  skal skrives ut. Standard: False
    UTDATA
    PS - Signalstyrke [W]
    
    EKSEMPEL
    # E-lag: Ne = 1e11, Te = Ti = 300 K, ki = 90,r = 100 km,dr = 1 km
    >>> signalpower(1e11,300,300,r=1e5,dr=1e3)
    1.8840080083693128e-18
    
    # F-lag: Ne = 1e12, Te = 2000 K, Ti = 250 K, ki = 90,r = 250 km,dr = 1 km
    >>> signalpower(1e12,2000,250,r=250e3,dr=1e3)
    6.695144007613304e-19
    """
    
    #Bølgelengde
    bl = c/f
    
    #Spredningstverrsnitt
    s = sxv(f,Ne,Te,Ti,ki,r,dr,theta,p)
    if p:
        if s.size < 10:
            print("Totalt spredningstverrsnitt:",s,"m²")
        else:
            print("Totalt spredningstverrsnitt:",s[0:10],"(...) m²")

    
    #Radarligninga
    PS = Pt*Gt*Gr*bl*bl*s/((4*pi)**3*r**4)
    
    return PS
    
    
def noisepower(B,TN):
    """
    Regner ut hvor sterk støyen er med funksjonen P_N = kB*T_sys*Br, der kB er 
    Boltzmanns konstant, T_sys er støytemperaturen til mottakeren og Br er 
    bandbredda til innkommende signal. 
    
    Calculates noise power
    
    INNDATA
    B   - Bandbredde til innkommende signal [Hz], kan være skalar eller np.array
    TN  - Støytemperatur [K], kan være skalar eller np.array, men må da ha samme størrelse som B. Standard: 100 K
    
    UTDATA
    PN  - støyeffekt [W], samme størrelse som B
    
    EKSEMPEL
    #Termisk ionehastighet
    >>> vt = vth(1000,16*amu)
    
    #Radarbølgetall
    >>> k = 2*pi/1.4 #1/m
    
    >>> noisepower(2*vt*k,100)
    8.934006914695719e-18
    
    """
    PN =  kB*TN*B
    return PN

def bandbredde(Ti,bl,dr,mi=16*amu):
    """
    Regner ut bandbredden til et signal, dvs. to ganger ionelinjebredde = 2*termisk ionehastighet*bølgetall
    Calculates the bandwidth of a signal, e.g. two times the width of the ion lines = 2*ion thermal velosity*wavenumber
    
    INNDATA
    Ti - ionetemperatur [K].
    bl - bølgelengde [m].
    dr - Tykkelse av spredningsvolumet [m].
    mi - ionemasse [kg]. Standard: 16 atommasseenheter (oksygen)
    
    UTDATA
    B  - Bandbredde [Hz]
    
    EKSEMPEL
    # E-lag: Ti = 300 K,bølgelengde = 1,4 m,dr = 1 km
    >>> bandbredde(Ti = 300,bl=1.4,dr=1e3)
    149896.229
    """
    
    #Bandbredda begrenses også av pulslengda tau_p = c/2dr
    #B = 2*vth(Ti,mi)*2*pi/bl
    B = max( 2*vth(Ti,32*amu)*2*pi/bl , c/(2*dr) )
    return B
    
def SNR(Ne,Te,Ti,ki=90,r=100e3,dr=1e3,f=c/1.4,theta=1,Gt=10**4.3,Gr=10**2.2,Pt=5e6,TN=100,B=[],p=False):
    """
    Regner ut signal-støy-forholdet til ei måling.
    Calculates signal-to-noise-ratio for a measurement
    
    INNDATA
    Ionosfære
    Ne - elektrontetthet [m⁻³].
    Te - elektrontemperatur [K].
    Ti - ionetemperatur [K].
    ki - spredningsvinkel [grader].         Standard:90 grader
    r  - avstand til spredningsvolumet [m]. Standard:100 km
    dr - Tykkelse av spredningsvolumet [m]. Standard:1 km
    Radar
    f  - Radarsendefrekvens [Hz].           Standard:21437470 Hz, tilsvarer 1,4 m
    theta- Strålebredde [grader].           Standard:1 grad
    Gt - Forsterkning til sender [].        Standard:10^4,3
    Gr - Forsterkning til mottaker [].      Standard:10^2,2
    Pt - Utsendt effekt [W].                Standard:5*10⁶ W
    
    B   - Bandbredde til innkommende signal [Hz]. Standard: 2*termisk ionehastighet
    TN  - Støytemperatur [K].               Standard: 100 K
    
    p   - Angir om verdier underveis skal skrives ut. Standard: False
    UTDATA
    SNR - Signal-støy-forhold []
    
    
    EKSEMPEL
    # E-lag: Ne = 1e11, Te = Ti = 300 K, ki = 90,r = 100 km,dr = 1 km
    >>> SNR(1e11,300,300,r=1e5,dr=1e3)
    0.3850133893101723
    
    # F-lag: Ne = 1e12, Te = 2000 K, Ti = 250 K, ki = 90,r = 250 km,dr = 1 km
    >>> SNR(1e12,2000,250,r=250e3,dr=1e3)
    0.14987998266713526
    
    """
    
    bl = c/f
    if B == []:
        
        #Bandbredda begrenses også av pulslengda tau_p = c/2dr
        #B = max( 2*vth(Ti,16*amu)*2*pi/bl , c/(2*dr) )
        B = bandbredde(Ti,bl,dr,mi=16*amu)
        
        
        
    s = signalpower(Ne,Te,Ti,ki,r,dr,f,theta,Gt,Gr,Pt,p)
    n = noisepower(B,TN)
    
    snr = s/n
    
    if p:
        print("Bandbredde:",B/1e3, "kHz")
        print("Signaleffekt:",s,"W")
        print("Støyeffekt:",n,"W")
    
    return snr

def msd(Ne,Te,Ti,ki=90,r=100e3,dr=1e3,f=c/1.4,theta=1,Gt=10**4.3,Gr=10**2.2,Pt=5e6,TN=100,B=[],K=[],mur=[],p=False):
    """
    Regner ut standardavviket/usikkerheta til ei måling.
    Calculates standard deviation/uncertanity for a measurement
    
    INNDATA
    Ionosfære
    Ne - elektrontetthet [m⁻³].
    Te - elektrontemperatur [K].
    Ti - ionetemperatur [K].
    ki - spredningsvinkel [grader].         Standard:90 grader
    r  - avstand til spredningsvolumet [m]. Standard:100 km
    dr - Tykkelse av spredningsvolumet [m]. Standard:1 km
    Radar
    f  - Radarsendefrekvens [Hz].           Standard:21437470 Hz, tilsvarer 1,4 m
    theta- Strålebredde [grader].           Standard:1 grad
    Gt - Forsterkning til sender [].        Standard:10^4,3
    Gr - Forsterkning til mottaker [].      Standard:10^2,2
    Pt - Utsendt effekt [W].                Standard:5*10⁶ W
    
    B   - Bandbredde til innkommende signal [Hz]. Standard: 2*termisk ionehastighet
    TN  - Støytemperatur [K].               Standard: 100 K
    
    Måledata. Må oppgis. Enten
    K   - Antall målinger []. 
    eller
    mur  - ønska relativ måleusikkerhet []
    Hvis begge er oppgitt, utgis relativ måleusikkerhet.
    
    p   - Angir om verdier underveis skal skrives ut. Standard: False
    UTDATA
    enten
    mu - Måleusikkerhet [W], hvis antall målinger er gitt
    eller
    K  - Antall målinger [], hvis måleusikkerhet er gitt, rundes opp til nærmeste heltall
    
    
    EKSEMPEL
    # E-lag: Ne = 1e11, Te = Ti = 300 K, ki = 90,r = 100 km,dr = 1 km,K=10
    # Signalstyrke
    >>> signalpower(1e11,300,300,r=1e5,dr=1e3)
    1.8840080083693128e-18
    >>> msd(1e11,300,300,r=1e5,dr=1e3,K=10)
    1.65814444609476e-18
    >>> msd(1e11,300,300,r=1e5,dr=1e3,mur = 0.05)
    3099
    
    # F-lag: Ne = 1e12, Te = 2000 K, Ti = 250 K, ki = 90,r = 250 km,dr = 1 km,K=10
    # Signalstyrke
    >>> signalpower(1e12,2000,250,r=250e3,dr=1e3)
    6.695144007613304e-19
    >>> msd(1e12,2000,250,r=250e3,dr=1e3,K=10)
    1.4283686296185776e-18
    >>> msd(1e12,2000,250,r=250e3,dr=1e3,mur=0.05)
    18207
    """
    
    bl = c/f
    if B == []:
        B = bandbredde(Ti,bl,dr,mi=16*amu)
        #B = 2*vth(Ti,16*amu)*2*pi/bl
    
    PS = signalpower(Ne,Te,Ti,ki,r,dr,f,theta,Gt,Gr,Pt,p)
    PN = noisepower(B,TN)
    
    # (PS² + PN²)
    enkeltvar = PS**2 + PN**2
    
    
    if K == []:
        if mur == []:
            raise TypeError('Verken antall målinger K eller ønska måleusikkerhet mur var oppgitt!')
        else:
            # Måleusikkerhet oppgitt. Da utgis antall nødvendige målinger
            ut = ceil(enkeltvar/(PS**2*mur**2))
    else:
         # Antall målinger oppgitt. Da utgis måleusikkerheta     
         ut = sqrt(enkeltvar/K)
         
         
    if p:
        print("Bandbredde:",B/1e3, "kHz")
        if PS.size<10:
            print("Signaleffekt:",PS,"W")
        else:
            print("Signaleffekt:",PS[0:10],"(...) W")
        print("Støyeffekt:",PN,"W")
         
    return ut

def inttime(Ne,Te,Ti,ki=90,r=100e3,dr=1e3,f=c/1.4,theta=1,Gt=10**4.3,Gr=10**2.2,Pt=5e6,TN=100,B=[],mur=0.05,tipp=2e-3,tpulse=0.5e-3,Np=1,mono=False,rmax=600e3,p=False,ElayerLongKorr=True):
    """
    Beregner hvor lang integrasjonstida må være for at resultatene får ønska nøyaktiughet
    Calculates how long the integration time must be to get results with desired accuracy
    
    
    INNDATA
    Ionosfære
    Ne - elektrontetthet [m⁻³].
    Te - elektrontemperatur [K].
    Ti - ionetemperatur [K].
    ki - spredningsvinkel [grader].         Standard:90 grader
    r  - avstand til spredningsvolumet [m]. Standard:100 km
    dr - Tykkelse av spredningsvolumet [m]. Standard:1 km
    Radar
    f  - Radarsendefrekvens [Hz].           Standard:21437470 Hz, tilsvarer 1,4 m
    theta- Strålebredde [grader].           Standard:1 grad
    Gt - Forsterkning til sender [].        Standard:10^4,3
    Gr - Forsterkning til mottaker [].      Standard:10^2,2
    Pt - Utsendt effekt [W].                Standard:5*10⁶ W
    
    B   - Bandbredde til innkommende signal [Hz]. Standard: 2*termisk ionehastighet
    TN  - Støytemperatur [K].               Standard: 100 K
    mur - ønska relativ måleusikkerhet [].  Standard: 0,05 = 5%
    tipp- Interpulsperiode (Tid mellom utsendelse av hver puls) [s]. Standard: 2 ms
    tpulse- Langpulslengde [s].     Standard: 0,5 ms
    ---------------------brukes ikke----Np  - Antall målinger per langpuls ()
    mono- om radaren er monostatisk.        Standard: False
    rmax- Lengste avtand med forventa signal (Kun for monostatisk radar) [m]. Standard = 600 km)
    
    p   - Angir om verdier underveis skal skrives ut. Standard: False
    ElayerLongKorr - Oppgir om korrelasjonstida er så lang at vi får Np*(Np-1)/2 målinger per langpuls Standard: True
    UTDATA
    it - integrasjonstid. [s]
    
    EKSEMPEL
    # E-lag: Ne = 1e11, Te = Ti = 300 K, ki = 90,r = 100 km,dr = 1 km,K=10
    >>> inttime(Ne=1e11,Te=300,Ti=300,f=c/1.4,theta=1,Gt=10**4.3,Gr=10**2.2,Pt=5e6,TN=100,mur=0.05,tipp=2e-3,tpulse=0.5e-3)
    9654.74
    
    """

    # Bølgelengde
    bl = c/f
    
    # Regner ut bandbredde
    if B == []:
        #Bandbredda begrenses også av pulslengda tau_p = c/2dr
         B = bandbredde(Ti,bl,dr,mi=16*amu)
         
    #Arbeidssyklus
    d = tpulse/tipp
        
    # Målefrekvens, dvs antall målinger per sekund
    # Measurement frequency
    
    #Begrensning av hvor ofte pulser sendes opp. (éi måling per langpuls)
    rm_p = 1/tipp
    
    #Begrensning av dekorrelasjonstid i måleområdet og tida det tar for én puls å komme seg gjennom måleområdet (=c/2dr)
#    rm_d = d*B # Var ikke korrekt
    rm_d = B
    
    # Begresnsning av monostatisk rader. Må vente til signalet fra fjerneste avstand er kommet tilbake.
    rm_r = c/(2*rmax)
    
    # Bitlengde, hvor lang en del av langpulsen er.
    tp = 2*dr/c
    
    # Antall biter i en langpuls
    Np = floor(tpulse/tp)
    
    # Antall målinger i en langpuls
    if ElayerLongKorr:
        mp = (Np-1)*Np/2
    else:
        mp = 1
    
    # Den størt mulige målefrekvensen blir da
    if mono:
        rm = min(rm_p,rm_d,rm_r)*mp

    else:
        rm = min(rm_p,rm_d)*mp
    
    #Så mange målinger trenger vi 
    K = msd(Ne,Te,Ti,ki,r,dr,f,theta,Gt,Gr,Pt,TN,B,mur=mur,p=p)
    
    # Integrasjonstid = nødvendige målinger/målefrekvens
    it = K/rm
    
    if p:
        if min(rm_p,rm_d,rm_r) == rm_d:
            print("Bandbredda satte begrensningene")
        elif min(rm_p,rm_d,rm_r) == rm_p:
            print("Pulsrepetisjonen satte begrensningene")
        elif min(rm_p,rm_d,rm_r) == rm_r:
            print("Begrensningene sattes av at radarstrålen måtte frem og tilbake")
            
        print("Største målefrekvens:",rm/1e3,"kHz")
        print("Langpulslengde:",tpulse*1e6,"µs")
        print("Delpulslengde:",tp*1e6,"µs")
        print("Antall målinger per langpuls",Np)
        if K.size<=10:
            print("Antall målinger:",K)
        else:
            print("Antall målinger:",K[0:10],"(...)")
    return it
    

def _plott_s0():
    
    
    #Eletrontemperatur
    Te = 2000 #K
    #Ionetemperatur
    Ti = 1000 #K
    #Elektrontetthet
    Ne = 3e11 # m⁻³
    #Spredningsvinkel
    ki = 90 #grader
    
    #Radarsendefrekvenser
    f = linspace(1e9,1000e9,1000) #1/s
    #Mulige bølgelengder til radaren
    bl = c/f
    
    D = debye(Ne,Te)
    s0 = sxv(f,Ne,Te,Ti,ki)
    
    #plt.plot
    plt.plot(bl,s0,label='Spredningstverrsnitt')
    plt.plot(array([D,D]),array([min(s0),max(s0)]),label='Debyelengde')
    #plt.ylim(0,max(s0))
    plt.xlabel("Bølgelengde [m]")
    plt.ylabel("Spredningstverrsnitt [m²]")
    plt.legend()
    
    
def _test(skriv_ut=False):
    import doctest
    doctest.testmod(verbose=skriv_ut)
#def get_settings(measurement):
#    if measurement == 'imaging':
#        ne = linspace(1e10,1e12,1000)
#        Te = 400
#        Ti = 300
#        ki = 90
#        r = 150e3
#        dr = [100,500,1000,1500,2000]
#        f = 230e6
#        theta = 1
#        mur = 0.05
#        Gt = 10**4.3
#        Gr = 10**2.2
#        Pt = 5e6
#        TN = 100
#        tipp = 2e-3
#        tpulse = 0.5e-3
#        mono = True
#        rmax = 600e3
##    return ne,Te,Ti,ki,r,dr,f,mur,Gt,Gr,TN,tipp,tpulse,mono,rmax
#    return ne,Te,Ti,ki,r,dr,f,theta,Gt,Gr,Pt,TN,[],mur,tipp,tpulse,1,mono,rmax
class Settings:
    def __init__(self,measurement):
        if measurement == 'imaging':
            self.ne = linspace(1e10,1e12,1000)
            self.Te = 400
            self.Ti = 300
            self.ki = 90
            self.r = 150e3
            self.dr = [100,500,1000,1500,2000]
            self.f = 233e6
            self.theta = 1
            self.mur = 0.05
            self.Gt = 10**4.3
            self.Gr = 10**2.2
            self.Pt = 5e6
            self.TN = 100
            self.tipp = 2e-3
            self.tpulse = 0.5e-3
            self.mono = True
            self.rmax = 600e3
        elif measurement == 'MIMO':
            self.ne = linspace(1e7,1e12,1000)
            self.Te = 180
            self.Ti = 180
            self.ki = 90
            self.r = 80e3
            self.dr = [100,500,1000,1500,2000]
            self.f = 230e6
            self.theta = 1
            self.mur = 0.05
            self.Gt = 10**4.3/3
            self.Gr = 10**4.3
            self.Pt = 5e6/3
            self.TN = 100
            self.tipp = 2e-3
            self.tpulse = 0.5e-3
            self.mono = True
            self.rmax = 600e3
        elif measurement == 'neD':
            self.ne = linspace(1e7,1e12,1000)
            self.Te = 180
            self.Ti = 180
            self.ki = 90
            self.r = 80e3
            self.dr = [100,500,1000,1500,2000]
            self.f = 230e6
            self.theta = 1
            self.mur = 0.05
            self.Gt = 10**4.3
            self.Gr = 10**4.3
            self.Pt = 5e6
            self.TN = 100
            self.tipp = 2e-3
            self.tpulse = 0.5e-3
            self.mono = True
            self.rmax = 300e3
        
    def resolution(self,p,xl = (5e10,1e12),ElayerLongKorr=True):
        if isinstance(self.ne,float):
#        it = inttime(ne,Te,Ti,ki,r,dr[0],f,mur = mur,Np=Np,p=True)
#            it = inttime(get_settings('imaging'),p)
            it = inttime(self.ne,self.Te,self.Ti,self.ki,self.r,self.dr,self.f,self.theta,\
                          self.Gt,self.Gr,self.Pt,self.TN,mur=self.mur,tipp=self.tipp,tpulse=self.tpulse,\
                         Np=1,mono=self.mono,rmax=self.rmax,p=p,ElayerLongKorr=ElayerLongKorr)
            print("Integrasjonstid:",it,"s")
        else:
            it = []
            ax = plt.axes()
            for i,res in enumerate(self.dr):
                it.append(inttime(self.ne,self.Te,self.Ti,self.ki,self.r,res,self.f,self.theta,\
                                  self.Gt,self.Gr,self.Pt,self.TN,mur=self.mur,tipp=self.tipp,tpulse=self.tpulse,\
                                  Np=1,mono=self.mono,rmax=self.rmax,p=p,ElayerLongKorr=ElayerLongKorr))
                ax.loglog(self.ne,it[i],label=("Range resolution: "+str(res)+" m"))
                print('')
            plt.grid()
            plt.xlabel("Electron density [m⁻³]")
            plt.ylabel("Integration time [s]")
            #plt.ylim(0,60)
#            plt.xlim(5e6,1e12)
            plt.xlim(xl)
            plt.legend()
            
    #        plt.savefig('Integrationtime.pdf')
            plt.show()
        

if __name__ == '__main__':
    settings = Settings('imaging')
    plt.figure(figsize=(7.5,5))
    settings.resolution(True)
#    plt.savefig('Integrationtime.pdf')
#    plt.show()
#    snr = SNR(Ne=1e11,Te=400.0,Ti=300.0,ki=90.0,r=150e3,dr=2e3,f=230e6,theta=1.0,Gt=10**4.3,Gr=10**2.3,Pt=5e6,p=True)
#    print("SNR:",snr)
#    
#    print(' ')
#    it = inttime(Ne=1e11,Te=400.0,Ti=300.0,ki=90.0,r=150e3,dr=2e3,f=230e6,mur=0.05,tipp=2e-3,tpulse=0.5e-3,Np=7,p=True)
#    print("Integrasjonstid:",it,"s")
    
    
#    ne = linspace(1e10,1e12,1000) # Burde inkludere alle muligheter
#    ne = linspace(1e6,1e12,1000) 
##    ne=1e11
#    Te = 400 # K
#    Ti = 300 # K
#    ki = 90  # grader
#    r = 80e3#150e3# m
#    dr = [100,500,1000,1500,2000] # m
##    dr = [1500]
#    f = 230e6 # Hz
#    mur = 0.05 #=5%
#    Np = 70 # Antall pulser
#    Gt = 10**4.3 # Senderforsterkning
#    Gr = 10**4.3#2.2 # Mottakerforsterkning
    
    
#    ne,Te,Ti,ki,r,dr,f,theta,Gt,Gr,Pt,TN,[],mur,tipp,tpulse,1,mono,rmax = get_settings('imaging')
#    settings = get_settings('imaging')    
#    ne = settings[0]
#    dr = settings[5]
#
#    if isinstance(ne,float):
##        it = inttime(ne,Te,Ti,ki,r,dr[0],f,mur = mur,Np=Np,p=True)
#        it = inttime(get_settings('imaging'),p=True)
#        print("Integrasjonstid:",it,"s")
#    else:
#        it = []
#        ax = plt.axes()
#        for i,res in enumerate(dr):
##            it.append(inttime(ne,Te,Ti,ki,r,res,f,Gt=Gt,Gr=Gr,mur = mur,Np=Np,p=True,Pt=5e6))
##            it.append(inttime(ne,Te,Ti,ki,r,res,f,theta,Gt,Gr,Pt,TN,[],mur,tipp,tpulse,1,mono,rmax,p=True))
#            it.append(inttime(settings,p=True))
#            ax.loglog(ne,it[i],label=("Range resolution: "+str(res)+" m"))
#            print('')
#        plt.grid()
#        plt.xlabel("Electron density [m⁻³]")
#        plt.ylabel("Integration time [s]")
#        #plt.ylim(0,60)
#        plt.xlim(5e6,1e12)
#        plt.legend()
#        
    
#        plt.show()
    #_test(skriv_ut=True)
    #_plott_s0()

    
    
    
    
    
