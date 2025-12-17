#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numpy import ndarray,ones,tan,pi,repeat,linspace,sqrt,arctan2,vstack,cos,sin
from numpy.matlib import repmat
from matplotlib.pyplot import scatter,title,xlabel,ylabel,contourf,colorbar,figure
from matplotlib.pyplot import show as show_plots
class Plasma:
    """ 
    Objekt som inneholder egenskapene til plasmaet som radaren ser
    Object that contains the properties of the plasma that the radar sees
    """
    def __init__(self,height,opening,coord='xy',res=(1,1),val=None,farfield=False,show=False,cut=False):
        """
        Lager plasma/radarmål i bestemt høyde og som fyller radarstrålen. 
        Antar sirkulær og perfekt radarstråle som peker loddrett opp (i senit). 
        Antallet målepunkter i plasmaet blir til sammen res[1]*res[2].
        
        
        Creates plasma/radar target at a certain height and filling the radar beam. 
        Assumes circular and perfect radar beam that points straight up (in zenit).
        The total number of measurement points in the plasma will be ang_res*rad_res.
        
        INNDATA
        height - høyde over radaren [m], skalar.
        opening - Åpningsvinkel til radaren [grader], skalar
        coord - Sier om plasmamålepunktene skal være fordelt polart (coord='pol') eller kartesisk (coord='xy'), se også nedenfor.
        val - Plasmaverdiene. Må ha samme størrelse som res oppgir
        farfield - sier om plasmaet er i fjernfeltet til radaren (Standard: False)
        res - Oppløsning i x- og y-retning/radiell og asimutal retning [heltall]
        show - (Valgfri) Angir om plasmaposisjonene skal vises med én gang. Standard: True

        
        UTDATA
        Plasma - objekt som tar vare på egenskapene til plasmaet
        
        coord = 'pol', res = (2,4):
                     x
               x            x
                     x
                   x   x
            x     x     x     x
                   x   x
                     x
               x            x
                     x      
            
        coord = 'xy', res = (4,4):
                  x     x     
            
            x     x     x     x           
            
            x     x     x     x
            
                  x     x     
        
        """
        
        if not isinstance(val,ndarray):
            val = ones(res)
        
        for g in res:
            if not isinstance(g,int):
                ValueError('Alle tallene i res må være heltall!')
            
            
            
        self.height = height
        self.opening = opening
        self.radius = height*tan(opening*pi/360)
        self.farfield = farfield
        
        if coord == 'pol':
            rad_res = res[0]
            ang_res = res[1]
            # Regner ut målepunkter i radiell og asimutal retning
            r = repeat(linspace(self.radius,0,rad_res+1)[0:-1],ang_res)
            teta = repmat(linspace(0,2*pi,ang_res+1)[0:-1],1,rad_res)
        
            # Setter sammen målepunkter til éi matrise
            # Collects the measurement points in one single matrix
            self.pos = vstack((r*cos(teta),r*sin(teta),repmat(self.height,1,ang_res*rad_res)))
            self.val = val.reshape(1,-1,order='F')
            self.rect = False
            
        elif coord == 'xy':            
            xres = res[0]
            yres = res[1]          
            x = repeat(linspace(-self.radius,self.radius,xres),yres).reshape(1,xres*yres)
            y = repmat(linspace(-self.radius,self.radius,yres),1,xres)
            
            
            self.pos = vstack([ x,y,repmat(self.height,1,x.shape[1]) ])
            self.val = val.reshape(1,-1,order='F')
            if res[0] == res[1]:
                self.rect = True
            else:
                self.rect = False
            
        if show:
            figure()
            self.show()
            
        
    def show(self,cmap=None,angles=True,at_once=False,cut=False):
        """ 
        Viser posisjonen til plasmamålepunktene projisert på horisontalplanet
        Shows horizontal distribution of the plasma measurement points
        """        
        if cut:
            # Hvilke punkter som skal vises
            show_point = sqrt(self.pos[0,]**2 + self.pos[1,]**2) <= self.radius
        else:
            show_point = ones(self.val.flatten().shape,dtype='bool')
        
        if angles:
            angx = 90 - arctan2(self.pos[2,show_point],self.pos[1,show_point]) * 180/pi
            angy = 90 - arctan2(self.pos[2,show_point],self.pos[0,show_point]) * 180/pi
#            angx = arctan2(self.pos[2,show_point],self.pos[1,show_point]) * 180/pi
#            angy = arctan2(self.pos[2,show_point],self.pos[0,show_point]) * 180/pi
            
            scatter(angx,angy,c = self.val.flatten()[show_point],cmap = cmap)
            ylabel('N/S Angle [degree]')
            xlabel('E/W Angle [degree]')
        else:
            scatter(self.pos[1,show_point],self.pos[0,show_point],c = self.val.flatten()[show_point],cmap = cmap)
            ylabel('North [m]')
            xlabel('East [m]')
        colorbar(label='Value')
        title('Plasma points and their values')
        if at_once:
            show_plots()

    def showestim(self,sol,t="",cmap=None,angles=False,cut=False,cb=True,cl=[]):
        """
        Viser løsning på inverst problem
        
        INNDATA
        sol - løsninga
        cmap - Fargeskjema - standard: standard for matplotlib
        angles - Om x- og y-aksen skal vise vinkler eller meter - standard: False/meter
        cut - om punkter utafor radarstrålken skal kuttes bort. Påtvinger at aksene er i meter - Standard: False
        """
        # min- og maksverdier til løsninga
        sM = max(max(sol))
        sm = min(min(sol))
        # Hvis z-aksen ikke er bestemt settes den slik at den går fra 0 til maksverdien hvis z-verdiene er positive eller til min- og maksverdien av z-aksen.
        if cl == []:
            if sM <= 0:
                cl = (sm,sM)
            if sol.max() > 0:
                cl = (0,max(max(sol)))
            else:
                cl = (0,1)
        levels = linspace(cl[0],cl[1],101)
        
        if cut:
            # Hvilke punkter som skal vises
            show_point = sqrt(self.pos[0,]**2 + self.pos[1,]**2) <= self.radius
            scatter(self.pos[1,show_point],self.pos[0,show_point],c = sol.flatten()[show_point],cmap = cmap)
            ylabel('North [m]')
            xlabel('East [m]')
        else:
            if angles:
                angx = 90 - arctan2(self.pos[2,],self.pos[1,]) * 180/pi
                angy = 90 - arctan2(self.pos[2,],self.pos[0,]) * 180/pi
                
                res = int(sqrt(angx.shape[0]))
    
                contourf(angx.reshape(res,res),angy.reshape(res,res),sol.flatten().reshape(res,res),levels,cmap = cmap)
                ylabel('N/S Angle [degree]')
                xlabel('E/W Angle [degree]')
            else:
                res = int(sqrt(self.pos.shape[1]))
                contourf(self.pos[1,].reshape(res,res),self.pos[0,].reshape(res,res),sol.flatten().reshape(res,res),levels,cmap = cmap)
                ylabel('North [m]')
                xlabel('East [m]')
        if cb:
            colorbar(label='Value')

        title(t)
