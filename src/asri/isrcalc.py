#!/usr/bin/env python3
"""
Funksjoner og klasser for å regne ut usikkerhet og integrasjonstid i målinger gjort med inkoherent spredningsradar.
"""
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from asri.plasma_funcs import debye, vth
from asri.konstanter import amu, c, kB, re

Kringkastbar = TypeVar('Kringkastbar', float, np.ndarray)

@dataclass
class ISR:
    """
    Dataklasse med egenskaper til radaren
    
    ```{py:attribute} Pt
    :type: float
    
    Sendereffekt [W]
    ```
    
    ```{py:attribute} Gt
    :type: float
    
    Senderforsterkning
    ```
    
    ```{py:attribute} Gr
    :type: float
    
    Mottakerforsterkning
    ```
    
    ```{py:attribute} f
    :type: float
    
    Senderfrekvens [Hz]
    ```
    ```{py:attribute} theta
    :type: float
    
    Strålebredde [°]
    
    ```{py:attribute} ki
    :type: float
    
    Spredningsvinkel [°]
    ```
    ```
    """
    Pt: float
    Gt: float
    Gr: float
    f: float
    theta: float
    ki: float = 90
    TN: float = 200
    
    @property
    def A_eff(self) -> float:
        "Effektivt mottakerareal [m²]"""
        return self.Gr*c*c/(self.f)**2
    
    def støyeffekt(self, fs: float) -> float:
        """
        Regn ut støyeffekt for bestemt samplingsfrekvens
        
        :param float fs: Samplingsfrekvens [Hz]
        :return: Støyeffekt [W]
        :rtype: float

        """
        return kB*self.TN*fs
@dataclass
class Plasmaparametre:
    """
    Dataklasser med egenskapene til et plasma
    
    ```{py:attribute} ne
    :type: float
    
    Elektrontetthet [m⁻³]
    ```
    ```{py:attribute} Te
    :type: float
    
    Elektrontemperatur [K]
    ```
    ```{py:attribute} Ti
    :type: float
    
    Ionetemperatur [K]
    ```
    ```{py:attribute} mi
    :type: float
    
    Gjennomsnittlig ionemasse [amu]
    ```
    """
    ne: float
    Te: float
    Ti: float
    mi: float
    
    def ionelinjebredde(self, f: float) -> float:
        """
        Regn ut bredda til ionelinja, dvs dobbel termiosk fart
        
        :param float f: Senderfrekvens [Hz]
        """
        k = 2*np.pi*c/f
        bb = 2*k*vth(self.Ti, self.mi*amu)
        return bb
        
    
def radarvolum(strålebredde: float, R: Kringkastbar, r: float) -> Kringkastbar:
    r"""
    Volum til radarstrålen for avstandsluka mellom R og R+r for stålebredda til radaren.
    
    ```{math}
    V = \frac{2\pi r}{3}\left(1-\cos\frac{\theta}{2}\right)\left(3R^2+3Rr+r^2\right)
    ```
    
    :param float strålebredde: Strålebredde [°]
    :param float | np.ndarray R: Avstand (til begynnelsen av avstandsluka)
    :param float r: Lengde til avstandsluke [m]
    :return: Radarvolum til gitte avstander [m³]
    :rtype: Kringkastbar (float | np.ndarray)


    ```{rubric} Eksempel
    ```
    Grensetilfelle: Regn ut volumet av ei kule
    ```python
    >>> import numpy as np
    >>> int(radarvolum(360, 0, 10/np.cbrt(np.pi)))
    1333
    
    ```
    """
    V = 2*np.pi*r/3 * (1-np.cos(np.deg2rad(strålebredde/2))) * (3*R*R + 3*R*r + r*r)
    return V

    
def mottatt_effekt(radar: ISR, plasmapar: Plasmaparametre, R: float, r:float) -> float:
    """
    Regn ut mottatt effekt gjennom inkoherent spredning i et spredt mål.
    
    :param ISR radar: Egenskaper til radaren
    :param Plasmaparametre plasmapar: Egenskaper til plasmaet
    :param float R: Avstand til (midten av) radarmålet (m)
    :param float r: Lengde til radarvolumet/acstandsluka [m]
    :return: Effekt mottatt av radaren [W]
    :rtype: float

    ```{rubric} Eksempel
    ```
    Eksempel for E-laget
    ```python
    >>> radar = ISR(5e6, 10**4.3, 10**2.2, 214e6, 1, 90)
    >>> pp = Plasmaparametre(1e11, 300, 300, 31)
    >>> float(mottatt_effekt(radar, pp, 100e3, 1e3))
    1.8669357677039284e-18
    
    ```
    Eksempel for F-laget
    ```python
    >>> radar = ISR(5e6, 10**4.3, 10**2.2, 214e6, 1, 90)
    >>> pp = Plasmaparametre(1e12, 2000, 250, 16)
    >>> float(mottatt_effekt(radar, pp, 250e3, 1e3))
    6.647023449707105e-19
    
    ```
    """
    # Pakk ut plasmaparametre for lettere handterig i funksjonen
    ne = plasmapar.ne
    Te = plasmapar.Te
    Ti = plasmapar.Ti
    
    alfa = 2*np.pi*debye(plasmapar.ne, plasmapar.Te)*radar.f/c
    thompsontverrsnitt = 4*np.pi*(re*np.sin(np.deg2rad(radar.ki)))**2
    brøk = 1 - 1/(1+alfa**2) + 1/( (1+alfa**2) * (1+alfa**2+Te/Ti) )
    V = radarvolum(radar.theta, R-r/2, r)
    
    spredningstverrsnitt = V*ne*brøk*thompsontverrsnitt
    avstandssvekkelse = 1/(R**4*(4*np.pi)**3)
    # mottatt effekt
    resultat = radar.Pt*radar.Gt*radar.A_eff*spredningstverrsnitt*avstandssvekkelse
    return resultat
    

def korrelasjonsusikkerhet(PS: float, PN: float, K: int)-> float:
    """
    Regn ut standardsvvik (usikkerhet) i én verdi av autokorrelasjonen
    
    :param float PS: Signaleffekt [W]
    :param float støyeffekt: Støyeffekt [W]
    :param int K: Antall målinger av autokorrelasjonen til signalet
    :return: Standardavvik i signalestimatet [W]
    :rtype: float
    
    ```{rubric} Eksempel
    ```
    Eksempel for E-laget
    ```python
    >>> radar = ISR(5e6, 10**4.3, 10**2.2, 214e6, 1, 90)
    >>> pp = Plasmaparametre(1e11, 300, 300, 31)
    >>> PS = mottatt_effekt(radar, pp, 100e3, 1e3)
    >>> float(korrelasjonsusikkerhet(PS, radar.støyeffekt(6e-6/6), 10))
    5.903769271179436e-19
    
    ```
    
    """
    return np.sqrt((PS**2 + PN**2)/K)

def integrasjonstid(PS: float, PN: float, ipp: float, relativ_måleusikkerhet: float) -> float:
    """
    Regn ut påkrevd integrasjonstid for at AKF-esimatene får ønska usikkerhet
    
    :param float PS: Signaleffekt [W]
    :param float støyeffekt: Støyeffekt [W]
    :param float ipp: Interpulsperiode [s]
    :param float relativ_måleusikkerhet: Ønska relativ måleusikkerhet
    :return: Integrasjonstid til ønska måleusikkerhet er oppnådd
    :rtype: float
    
    ```{rubric} Eksempel
    ```
    Eksempel for E-laget
    ```python
    >>> radar = ISR(5e6, 10**4.3, 10**2.2, 214e6, 1, 90)
    >>> pp = Plasmaparametre(1e11, 300, 300, 31)
    >>> PS = mottatt_effekt(radar, pp, 100e3, 1e3)
    >>> float(integrasjonstid(PS, radar.støyeffekt(20e3), 2*600e3/c, 0.05))
    1402.7384237931697
    
    ```
    """
    K = int(np.ceil((PS**2 + PN**2)/(PS**2*relativ_måleusikkerhet**2)))
    tid = K*ipp
    return tid
    
    

if __name__ == '__main__':
    import doctest
    doctest.testmod()