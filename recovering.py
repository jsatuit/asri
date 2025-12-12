#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from radaro3 import Radar,Timer,svddiag,savefig,clf,close,fillagresti,loglog,xlabel,\
ylabel,plot,title,subplot,legend,semilogy,show_plots,trp1,trp2,figure,File,asarray,ylim,show_layout_plots,alfabetet
#import h5py
from plasma import Plasma
from numpy import zeros,nan,sqrt,array,dtype,ndarray,nanmedian
import numpy
from datetime import datetime#, datetime.now() gir ut nåværende klokkeslett
import os
from matplotlib.pyplot import rcParams

def bytes2str(txt):
    if type(txt) == ndarray:
        return [str(n)[2:-1] for n in txt]
    elif type(txt) == numpy.bytes_:
        return str(txt)[2:-1]
def reccalc(layouts,target,sigma=0.05,h=100e3,o=1,N=[18,20],methods=['MKM'],methodparams=[None],farfield=False,showres=True,showsds=True,normcomp=False):
    """
    Går gjennom ulike oppsett og antatte bildestørrelser og måler usikkerheta til resultatet.
    Looprs over different layouts and imaging resolutions and measures uncertainty of the result
    
    methods - må være liste eller 'svddiag'
    - must be a list or 'svddiag' (if the latter, svddiag is called for the imaging, else Radar.comprec is called)

    """
    # Tidspunktet for når beregningene starta
    tp = datetime.now()
    with Timer('Preallokering'):
        if isinstance(N,int):
            N = [N,N]
        
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
        with Timer(('runde'+str(n))):
            res = (n,n)
            plasma = Plasma(height=h,opening=o,coord='xy',res=res,val=None,show=False,cut=False,farfield=farfield)
            for nl,layout in enumerate(layouts):
                radar = Radar(layout=layout)
                m,x = radar.get_meas(target,farfield=farfield,sigma=sigma)

                # Bruk målingene
                if farfield:
                    A,vi,vf = radar.theorymatrix(plasma,inv_comp=True)
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
                    print('Sigma er ',SIGMA,'og settes derfor til nan')
                    SIGMA = nan
#                SIGMA[SIGMA <0] = nan
                # Beregn og plott kovarians / standardavvik
                S = radar.basiscov(plasma,sd=SIGMA,show_cov=False,show_sd=showsds)
                # Standardavvik
#                s = sqrt(S.real.diagonal().reshape(res))
#                smean = s.mean()
                smean = sqrt(S.real.diagonal().mean())
                if not smean > 0:
                    txst = ('Her har det skjedd noe skikkelig galt. (Den positive) kvadratrota av et reelt tall,'+str(S.real.diagonal().mean())+', ble negativ,'+str(smean), '!')
#                    print('Varianser til MKM-estimater',S.real.diagonal().reshape(res))
                    print(txst)
#                    raise RuntimeError(txst)
                # Midlere standardavvik
                sm[nl,n-N[0]] = smean
                shatt[nl,n-N[0]] = SIGMA
                sse[nl,n-N[0]] = SSE
                mse[nl,n-N[0],:] = array(sim)
#                filnavn = fillagresti+'/Results/L-' + str(radar.layout) + '_' + 'rad-' + str(round(plasma.radius)) + \
#                '_hkm-'+str(round(plasma.height/1e3)) + '_ps-'+str(n)+'x'+str(n) + 
                
    
    filnavn = fillagresti+'/Results/'+str(tp)[0:-7]+".hdf5"
    with File(filnavn, "w") as f:
#        dt = h5py.special_dtype(vlen=str)   # For å lagre strenger
        if not isinstance(methods,list):
            f.create_dataset("MidlStdAvSVD", data=sSVD )
            f.create_dataset("MidlStdAvTkh1", data=sT21 )
            f.create_dataset("MidlStdAvTkh10", data=sT210 )
            f.create_dataset("methods", data=methods)
            f.create_dataset("methodparams", data=methodparams )
        else:
            f.create_dataset("methods", data=[n.encode("ascii", "ignore") for n in methods])
            f.create_dataset("methodparams", data=[str(n).encode("ascii", "ignore") for n in methodparams])
        f.create_dataset("layouts", data=[str(n).encode("ascii", "ignore") for n in layouts])
#        f.create_dataset("xrec", data=xrec )
        f.create_dataset("MidlStdAv", data=sm )
        f.create_dataset("shatt", data=shatt )

        f.create_dataset("SSE", data=sse )
        f.create_dataset("Ulikhet", data=mse )
        f.create_dataset("TikhRegParam1", data=trp1 )
        f.create_dataset("TikhRegParam2", data=trp2 )
        f.create_dataset("StøyStdAv", data=sigma )
        f.create_dataset("Nv", data=Nv )
def plot_results(filnavn,forartikkel=False,smallest_measurement=0.1,dobbelplott=False,midlukj=0.5): # Målinger mer i [1/m³] (¡sukk!)
    with Timer('Plotting'):
        
        with File(filnavn,"r") as f:
            methods = asarray(f["methods"])
            methodparams = asarray(f["methodparams"])
            layouts = bytes2str(asarray(f["layouts"]))
            print("Layouts:", layouts)
            Nv = asarray(f["Nv"]) # Piksler i hver retning
            sm = asarray(f["MidlStdAv"]) # Dimensjonsløs (må ganges med minste måling)
            shatt = asarray(f["shatt"]) # Dimensjonsløs
#            if not isinstance(methods,'|S21'):
            if not methods.dtype == dtype("S5"):
#                print(methods,methods.dtype)
                methods = str(methods)
                methodparams = float(methodparams)
                sSVD = asarray(f["MidlStdAvSVD"]) # Dimensjonsløs (må ganges med minste måling)
                sT21 = asarray(f["MidlStdAvTkh1"]) # Dimensjonsløs (må ganges med minste måling)
                sT210 = asarray(f["MidlStdAvTkh10"]) # Dimensjonsløs (må ganges med minste måling)
            else:
                methods = bytes2str(methods)
                methodparams = bytes2str(methodparams)
            sse = asarray(f["SSE"])
            mse = asarray(f["Ulikhet"]) # Dimensjon til sanne verdier ^2
            trp1 = asarray(f["TikhRegParam1"])
            trp2 = asarray(f["TikhRegParam2"])
            sigma = asarray(f["StøyStdAv"]) # Dimensjonsløs
            try:
                xrec = asarray(f["xrec"])
            except:
                print('Finner ikke xrec')
        if dobbelplott:
            print("Valgte dobbelplott")
            farger = (rcParams['axes.prop_cycle']).by_key()['color']
            
            nlayouts = len(layouts)
            figure(figsize=(16,4*nlayouts))
            
            for nl in range(len(layouts)//2):
                nl *= 2
#                figure(figsize = (15,6))
#                subplot(1,2,1)
                subplot(nlayouts//2,2,nl+1)
                if type(methods) == list:
                    plot(array(Nv),sm[nl,:]/midlukj,lw=2,label=layouts[nl])
                    plot(array(Nv),sm[nl+1,:]/midlukj,'--',label=layouts[nl+1])
                else:
                    loglog(array(Nv),sm[nl,:]*sigma/shatt[nl,:]*Nv*Nv*smallest_measurement/midlukj,label=layouts[nl]+' LS',lw=2,color=farger[0])
                    loglog(array(Nv),sqrt(sSVD[nl,:]) *Nv*Nv*smallest_measurement/midlukj,label=layouts[nl]+' TSVD, rcond='+str(methodparams),lw=2,color=farger[1])
                    loglog(array(Nv),sqrt(sT21[nl,:])*Nv*Nv*smallest_measurement /midlukj,label=layouts[nl]+' T2, alpha='+str(trp1),lw=2,color=farger[2])
                    loglog(array(Nv),sqrt(sT210[nl,:])*Nv*Nv*smallest_measurement/midlukj,label=layouts[nl]+' T2, alpha='+str(trp2),lw=2,color=farger[3])
                    loglog(array(Nv),sm[nl+1,:]*sigma/shatt[nl+1,:]*Nv*Nv*smallest_measurement/midlukj,'--',label=layouts[nl+1]+' LS',color=farger[0])
                    loglog(array(Nv),sqrt(sSVD[nl+1,:]) *Nv*Nv*smallest_measurement/midlukj,'--',label=layouts[nl+1]+' TSVD, rcond='+str(methodparams),color=farger[1])
                    loglog(array(Nv),sqrt(sT21[nl+1,:])*Nv*Nv*smallest_measurement /midlukj,'--',label=layouts[nl+1]+' T2, alpha='+str(trp1),color=farger[2])
                    loglog(array(Nv),sqrt(sT210[nl+1,:])*Nv*Nv*smallest_measurement/midlukj,'--',label=layouts[nl+1]+' T2, alpha='+str(trp2),color=farger[3])
                title("("+alfabetet[nl]+') Mean standard deviation s²(A''A)⁻¹')
                ylabel('Relative pixelintensity')
                xlabel('Assumed resolution [px]')
                ylim(smallest_measurement*1e-3/midlukj,smallest_measurement*1e3/midlukj)
                legend()
                
#                subplot(1,2,2)
                subplot(nlayouts//2,2,nl+2)
                if type(methods) == list:
                    for nm,method in enumerate(methods):
                        semilogy(array(Nv),mse[nl,:,nm]/midlukj,lw=2,label=(layouts[nl]+' '+method))
                        semilogy(array(Nv),mse[nl+1,:,nm]/midlukj,'--',label=(layouts[nl+1]+' '+method))
                elif methods == 'svddiag':
                    semilogy(array(Nv),sqrt(mse[nl,:,0])/midlukj,label=(layouts[nl]+' '+'MKM'),lw=2,color=farger[0])
                    semilogy(array(Nv),sqrt(mse[nl,:,1])/midlukj,label=(layouts[nl]+' '+'TSVD, rcond='+str(methodparams)),lw=2,color=farger[1])
                    semilogy(array(Nv),sqrt(mse[nl,:,2])/midlukj,label=(layouts[nl]+' '+'Tikhonov alpha='+str(trp1)),lw=2,color=farger[2])
                    semilogy(array(Nv),sqrt(mse[nl,:,3])/midlukj,label=(layouts[nl]+' '+'Tikhonov alpha='+str(trp2)),lw=2,color=farger[3])
                    semilogy(array(Nv),sqrt(mse[nl+1,:,0])/midlukj,'--',label=(layouts[nl+1]+' '+'MKM'),color=farger[0])
                    semilogy(array(Nv),sqrt(mse[nl+1,:,1])/midlukj,'--',label=(layouts[nl+1]+' '+'TSVD, rcond='+str(methodparams)),color=farger[1])
                    semilogy(array(Nv),sqrt(mse[nl+1,:,2])/midlukj,'--',label=(layouts[nl+1]+' '+'Tikhonov alpha='+str(trp1)),color=farger[2])
                    semilogy(array(Nv),sqrt(mse[nl+1,:,3])/midlukj,'--',label=(layouts[nl+1]+' '+'Tikhonov alpha='+str(trp2)),color=farger[3])
                else:
                    semilogy(array(Nv),mse[nl,:,:]/midlukj,lw=2,label=layouts[nl])
                    semilogy(array(Nv),mse[nl+1,:,:]/midlukj,'--',label=layouts[nl+1])
                title("("+alfabetet[nl+1]+') Mean sq. error (diff. from orig., scal. image for comp.)')
                ylabel('Relative pixel intensity')
                ylim(nanmedian(mse[nl,:,:].flatten())*5e-1/midlukj,nanmedian(mse[nl,:,:].flatten())*1e3/midlukj)
                xlabel('Assumed resolution [px]')
                legend()
#                navn = fillagresti+'plots/'+layouts[nl]+'_errors.png'
            navn = fillagresti+'plots/'+filnavn.rsplit("/")[-1][:-5]+'_errors.png'
            savefig(navn)
            show_plots()
        else:
            for nl,layout in enumerate(layouts):
                
                if forartikkel:
                    figure(figsize = (15,6))
                    subplot(1,2,1)
                else:
                    figure(figsize = (15,12))
                    subplot(2,2,1)
                if type(methods) == list:
                    plot(array(Nv),sm[nl,:],label=layout)
                else:
                    try:
                        loglog(array(Nv),sm[nl,:]*sigma/shatt[nl,:]*Nv*Nv*smallest_measurement,label=layout+' LS')
                    except:
                        print('Kunne ikke plotte variansen til MKM-estimatet!')
                        print(array(Nv),sm[nl,:])
                    try:
                        loglog(array(Nv),sqrt(sSVD[nl,:]) *Nv*Nv*smallest_measurement,label=layout+' TSVD, rcond='+str(methodparams))
                    except:
                        print('Kunne ikke plotte variansen til TSVD-estimatet!')
                    try:
                        loglog(array(Nv),sqrt(sT21[nl,:])*Nv*Nv*smallest_measurement ,label=layout+' T2, alfa='+str(trp1))
                    except:
                        print('Kunne ikke plotte variansen til T2_1-estimatet!')
                    try:
                        loglog(array(Nv),sqrt(sT210[nl,:])*Nv*Nv*smallest_measurement,label=layout+' T2, alfa='+str(trp2))
                    except:
                        print('Kunne ikke plotte variansen til T2_1-estimatet!')
                title('Mean standard deviation s²(A''A)⁻¹')
                ylabel('Pixelintensity [1/m³]')
                xlabel('Assumed resolution [px]')
                ylim(smallest_measurement*1e-3,smallest_measurement*1e3)
                legend()
                if not forartikkel:
                    subplot(2,2,4)
                    plot(array(Nv),shatt[nl,:],label=layout)
                    title('SD (measurement), estimated')
                    ylabel('Power [W]')
                    xlabel('Assumed resolution [px]')
                    legend()
                    subplot(2,2,3)
                    plot(array(Nv),sse[nl,:],label=layout)
                    title('SSE')
                    ylabel('Power [W]')
                    xlabel('Assumed resolution [px]')
                    legend()    
                
                    subplot(2,2,2)    
                else:
                    subplot(1,2,2)
                if type(methods) == list:
                    for nm,method in enumerate(methods):
                        semilogy(array(Nv),mse[nl,:,nm],label=(layout+' '+method))
                elif methods == 'svddiag':
                    semilogy(array(Nv),sqrt(mse[nl,:,0]),label=(layout+' '+'MKM'))
                    semilogy(array(Nv),sqrt(mse[nl,:,1]),label=(layout+' '+'TSVD, rcond='+str(methodparams)))
                    semilogy(array(Nv),sqrt(mse[nl,:,2]),label=(layout+' '+'Tikhonov alfa='+str(trp1)))
                    semilogy(array(Nv),sqrt(mse[nl,:,3]),label=(layout+' '+'Tikhonov alfa='+str(trp2)))
                else:
                    semilogy(array(Nv),mse[nl,:,:],label=layout)
                title('Mean sq. error (diff. from orig., scal. image for comp.)')
                ylabel('Pixel intensity [1/m³]')
                ylim(nanmedian(mse[nl,:,:].flatten())*5e-1,nanmedian(mse[nl,:,:].flatten())*1e3)
                xlabel('Assumed resolution [px]')
                legend()
    #            if showcovplot == 'save':
                navn = fillagresti+'plots/'+layout+'_errors.png'
                savefig(navn)
    #                clf()
    #                close('all')
    #            else:
                show_plots()
                    
                    
                    
if __name__ == '__main__':
#    layouts = ['singtranscore','singtrans','3transcore','3trans']
#    layouts = ['3trans']
#    layouts= ['singtranscore']
#    layouts = ['kvadrat','MBS','T2']

#    with Timer('Oppsett'):
#        show_layout_plots(layouts,farfield=True)
#    with Timer('Rec_plots'):
#        reccalc(layouts,'nordlys',0.05,100e3,1,[8,9],['MKM','TSVD','capon'],[None,2e-2,None],False,True,False)
#        reccalc(layouts,'nordlys',0.0,100e3,1,[6,7],'svddiag',2e-2,False,True,showsds=True,normcomp=False)
        
#        reccalc(layouts,'prikk129',0.05,100e3,1,[8,8],['MF','MKM','TSVD','capon'],[None,None,2e-2,None],False,True,False)
        filsti = fillagresti+"Results"
        filliste = os.listdir(filsti)
        filliste.sort()
#        plot_results(fillagresti+"Results/"+filliste[-7],forartikkel=True,dobbelplott=True)
        plot_results(fillagresti+"Results/"+filliste[-1],forartikkel=True,dobbelplott=True)
        
        
        
        
        
        
        
        
        
        
        
        
        
