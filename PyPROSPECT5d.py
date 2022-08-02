import numpy as np
import scipy.special as sc
from lmfit import Minimizer, Parameters
from abs_coefs import PDB_absorption_coefficients
from tav import calctav
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import gc

def inversion5D_reflectance_transmittance(wl,refl,trans,params,savestr='',plot=False):
    
    #Minimization function for inversion
    def RMSE_minimization(params,WL,refl,trans): 
        N     = params['N']
        Cab   = params['Cab']
        Car   = params['Car']
        Ant   = params['Ant']
        Brown = params['Brown']
        Cw    = params['Cw']
        Cm    = params['Cm']
        
        LRT  = Prospect5D(WL,N,Cab,Car,Ant,Brown,Cw,Cm)
        
        return (np.append(LRT[:,1]-refl,LRT[:,2]-trans))
    
    lmParams = Parameters()
        
    #Set default values and bounds for Prospect5 parameters.
    #If any parameter values are provided, set those parameters as fixed values.
    if params['N'] != None:
        lmParams.add('N',value=params['N'],vary=False)
    else:
        lmParams.add('N',value=1.2,min=0.2,max=5)
 
    if params['Cab'] != None:
        lmParams.add('Cab',value=params['Cab'],vary=False)
    else:
        lmParams.add('Cab',value=30,min=2,max=150)          

    if params['Car'] != None:
        lmParams.add('Car',value=params['Car'],vary=False)
    else:
        lmParams.add('Car',value=20,min=0,max=50)   
        
    if params['Ant'] != None:
        lmParams.add('Ant',value=params['Ant'],vary=False)
    else:
        lmParams.add('Ant',value=1,min=0,max=60)   

    if params['Brown'] != None:
        lmParams.add('Brown',value=params['Brown'],vary=False)
    else:
        lmParams.add('Brown',value=1,min=0,max=50) 

    if params['Cw'] != None:
        lmParams.add('Cw',value=params['Cw'],vary=False)
    else:
        lmParams.add('Cw',value=0.015,min=0.00001,max=0.5) 

    if params['Cm'] != None:
        lmParams.add('Cm',value=params['Cm'],vary=False)
    else:
        lmParams.add('Cm',value=0.009,min=0.00001,max=0.5) 

    #Optimize data based on lmfit minimizer. 
    #Use trust region reflective method e.g. Jay et al., 2016
    opt_function = Minimizer(RMSE_minimization, lmParams, fcn_args=(wl,refl,trans))
    opt          = opt_function.minimize('least_squares')
    
    #Combine optimized params into singular array, calculate LRT and RMSE of fit    
    opt_params = np.array((opt.params['N'].value,opt.params['Cab'].value,opt.params['Car'].value,
                           opt.params['Ant'].value,opt.params['Brown'].value,opt.params['Cw'].value,
                           opt.params['Cm'].value)) 
    
    LRT        =  Prospect5D(wl,opt_params[0],opt_params[1],opt_params[2],opt_params[3],
                             opt_params[4],opt_params[5],opt_params[6])    
    opt_params = np.append(opt_params,np.sqrt(np.mean(np.append(LRT[:,1]-refl,LRT[:,2]-trans)**2))) #RMSE   

    #Plot observed and modeled reflectance is plotting is turned on.
    if (plot==True):
        
        fig,ax = plt.subplots(3,1,figsize=(12.5,18))
        ax[0].plot(wl,refl,lw=3,ls='-',c='black')
        ax[0].plot(wl,LRT[:,1],lw=3,ls='--',c='#9b9b9b')
        ax[0].set_xlim(wl[0],wl[-1])
        ax[0].set_xticks(np.linspace(wl[0],wl[-1],5))
        ax[0].set_xticklabels(np.linspace(wl[0],wl[-1],5),fontsize=26)
        ax[0].set_xlabel('Wavelength (nm)',fontsize=28)
        ax[0].set_ylim(0,1)
        ax[0].set_yticks([0,.25,.5,.75,1])
        ax[0].set_yticklabels([0,.25,.5,.75,1],fontsize=26)
        ax[0].set_ylabel('Reflectance',fontsize=28)
        ax[0].tick_params('both', length=10, width=1.25, which='major',labelsize=24)
        ax[0].annotate('RMSE: %.4f'%np.sqrt(np.mean((LRT[:,1]-refl)**2)), xy=(.025,.77),xycoords='axes fraction',fontsize=28)
        for axis in ['top','bottom','left','right']:
           ax[0].spines[axis].set_linewidth(1.25)
           
        ax[1].plot(wl,trans,lw=3,ls='-',c='black')
        ax[1].plot(wl,LRT[:,2],lw=3,ls='--',c='#9b9b9b')
        ax[1].set_xlim(wl[0],wl[-1])
        ax[1].set_xticks(np.linspace(wl[0],wl[-1],5))
        ax[1].set_xticklabels(np.linspace(wl[0],wl[-1],5),fontsize=26)
        ax[1].set_xlabel('Wavelength (nm)',fontsize=28)
        ax[1].set_ylim(0,1)
        ax[1].set_yticks([0,.25,.5,.75,1])
        ax[1].set_yticklabels([0,.25,.5,.75,1],fontsize=26)
        ax[1].set_ylabel('Transmittance',fontsize=28)
        ax[1].tick_params('both', length=10, width=1.25, which='major',labelsize=24)
        ax[1].annotate('RMSE: %.4f'%np.sqrt(np.mean((LRT[:,2]-trans)**2)), xy=(.025,.77),xycoords='axes fraction',fontsize=28)
        for axis in ['top','bottom','left','right']:
           ax[1].spines[axis].set_linewidth(1.25)
           
        ax[2].plot(wl,1-(refl+trans),lw=3,ls='-',c='black')   
        ax[2].plot(wl,1-(LRT[:,1]+LRT[:,2]),lw=3,ls='--',c='#9b9b9b')
        ax[2].set_xlim(wl[0],wl[-1])
        ax[2].set_xticks(np.linspace(wl[0],wl[-1],5))
        ax[2].set_xticklabels(np.linspace(wl[0],wl[-1],5),fontsize=26)
        ax[2].set_xlabel('Wavelength (nm)',fontsize=28)
        ax[2].set_ylim(0,1)
        ax[2].set_yticks([0,.25,.5,.75,1])
        ax[2].set_yticklabels([0,.25,.5,.75,1],fontsize=26)
        ax[2].set_ylabel('Absorbance',fontsize=28)
        ax[2].tick_params('both', length=10, width=1.25, which='major',labelsize=24)
        ax[2].annotate('RMSE: %.4f'%np.sqrt(np.mean(((1-(LRT[:,1]+LRT[:,2]))-(1-(refl+trans)))**2)), xy=(.025,.77),xycoords='axes fraction',fontsize=28)
        for axis in ['top','bottom','left','right']:
           ax[2].spines[axis].set_linewidth(1.25)
           
        plt.tight_layout()
        fig.savefig(savestr,dpi=100)
        plt.cla()
        plt.clf()
        plt.close('all')
        plt.close(fig)
        gc.collect()
    
    return opt_params,LRT    
    
def inversion5D_reflectance(wl,refl,params,trans=None,savestr='',plot=False):
    '''
    Invert the Prospect5D model using reflectance data only. Inversion uses a 
    least-squares curve fit with a trust region reflective algorithm
    (see Jay et al., 2016, Hill et al., 2019)

    Parameters
    ----------
    WL : array_like
        Array of wavelengths. Valid wavelengths range from 400 nm - 2500 nm.
    refl : array_like
        Leaf reflectance to be inverted. Must have the same length as WL.
    params : dict
        Dict of Prospect5D parameters. Must contain:
            
            N     : NONE or float
            Cab   : NONE or float
            Car   : NONE or float
            Ant   : NONE or float
            Brown : NONE or float
            Cw    : NONE or float
            Cm    : NONE or float
        
        Each parameter should be assigned NONE, unless you want assign a
        particular value - if one or more parameters has been independently
        measured, for example.
    savestr : str, optional
        DESCRIPTION. The default is ''.
    plot : boolean, optional
        If set to TRUE, create a plot of modeled and observed reflectance.
        Setting to TRUE requires a real value for savestr.

    Returns
    -------
    opt_params : array_like
        Optimized parameter values and fitting RMSE. 
        [N,Cab,Car,Ant,Brown,Cw,Cm,RMSE]
    LRT : array_like
        Array containing WL, modeled reflectance, and modeled transmittance.
        
    Examples
    --------
    >>> WL = np.arange(400,2501,1)
    >>> params = {'N':None,'Cab':None,'Car':None,'Ant':None,
                  'Brown':None,'Cw':None,'Cm':None}
    >>> coefficients, LRT = inversion5D_reflectance(WL,refl,params)
    '''
    
    #Minimization function for inversion
    def RMSE_minimization(params,WL,refl): 
        N     = params['N']
        Cab   = params['Cab']
        Car   = params['Car']
        Ant   = params['Ant']
        Brown = params['Brown']
        Cw    = params['Cw']
        Cm    = params['Cm']
        
        LRT  = Prospect5D(WL,N,Cab,Car,Ant,Brown,Cw,Cm)
        return (LRT[:,1]-refl)
    
    lmParams = Parameters()
        
    #Set default values and bounds for Prospect5 parameters.
    #If any parameter values are provided, set those parameters as fixed values.
    if params['N'] != None:
        lmParams.add('N',value=params['N'],vary=False)
    else:
        lmParams.add('N',value=1.2,min=0.2,max=5)
 
    if params['Cab'] != None:
        lmParams.add('Cab',value=params['Cab'],vary=False)
    else:
        lmParams.add('Cab',value=30,min=2,max=150)          

    if params['Car'] != None:
        lmParams.add('Car',value=params['Car'],vary=False)
    else:
        lmParams.add('Car',value=20,min=0,max=50)   
        
    if params['Ant'] != None:
        lmParams.add('Ant',value=params['Ant'],vary=False)
    else:
        lmParams.add('Ant',value=1,min=0,max=60)   

    if params['Brown'] != None:
        lmParams.add('Brown',value=params['Brown'],vary=False)
    else:
        lmParams.add('Brown',value=1,min=0,max=50) 

    if params['Cw'] != None:
        lmParams.add('Cw',value=params['Cw'],vary=False)
    else:
        lmParams.add('Cw',value=0.015,min=0.00001,max=0.5) 

    if params['Cm'] != None:
        lmParams.add('Cm',value=params['Cm'],vary=False)
    else:
        lmParams.add('Cm',value=0.009,min=0.00001,max=0.5) 

    #Optimize data based on lmfit minimizer. 
    #Use trust region reflective method e.g. Jay et al., 2016
    opt_function = Minimizer(RMSE_minimization, lmParams, fcn_args=(wl, refl))
    opt          = opt_function.minimize('least_squares')
    
    #Combine optimized params into singular array, calculate LRT and RMSE of fit    
    opt_params = np.array((opt.params['N'].value,opt.params['Cab'].value,opt.params['Car'].value,
                           opt.params['Ant'].value,opt.params['Brown'].value,opt.params['Cw'].value,
                           opt.params['Cm'].value)) 
    
    LRT        =  Prospect5D(wl,opt_params[0],opt_params[1],opt_params[2],opt_params[3],
                             opt_params[4],opt_params[5],opt_params[6])    
    opt_params = np.append(opt_params,np.sqrt(np.mean((LRT[:,1]-refl)**2))) #RMSE
    

    #Plot observed and modeled reflectance is plotting is turned on.
    if (plot==True):
        fig,ax = plt.subplots(3,1,figsize=(12.5,18))
        ax[0].plot(wl,refl,lw=3,ls='-',c='black')
        ax[0].plot(wl,LRT[:,1],lw=3,ls='--',c='#9b9b9b')
        ax[0].set_xlim(wl[0],wl[-1])
        ax[0].set_xticks(np.linspace(wl[0],wl[-1],5))
        ax[0].set_xticklabels(np.linspace(wl[0],wl[-1],5),fontsize=26)
        ax[0].set_xlabel('Wavelength (nm)',fontsize=28)
        ax[0].set_ylim(0,1)
        ax[0].set_yticks([0,.25,.5,.75,1])
        ax[0].set_yticklabels([0,.25,.5,.75,1],fontsize=26)
        ax[0].set_ylabel('Reflectance',fontsize=28)
        ax[0].tick_params('both', length=10, width=1.25, which='major',labelsize=24)
        ax[0].annotate('RMSE: %.4f'%np.sqrt(np.mean((LRT[:,1]-refl)**2)), xy=(.025,.77),xycoords='axes fraction',fontsize=28)
        for axis in ['top','bottom','left','right']:
           ax[0].spines[axis].set_linewidth(1.25)
           
        ax[1].plot(wl,LRT[:,2],lw=3,ls='--',c='#9b9b9b')
        ax[1].set_xlim(wl[0],wl[-1])
        ax[1].set_xticks(np.linspace(wl[0],wl[-1],5))
        ax[1].set_xticklabels(np.linspace(wl[0],wl[-1],5),fontsize=26)
        ax[1].set_xlabel('Wavelength (nm)',fontsize=28)
        ax[1].set_ylim(0,1)
        ax[1].set_yticks([0,.25,.5,.75,1])
        ax[1].set_yticklabels([0,.25,.5,.75,1],fontsize=26)
        ax[1].set_ylabel('Transmittance',fontsize=28)
        ax[1].tick_params('both', length=10, width=1.25, which='major',labelsize=24)       
        ax[1].plot(wl,LRT[:,2],lw=3,ls='-',c='black')
        for axis in ['top','bottom','left','right']:
           ax[1].spines[axis].set_linewidth(1.25)
           
        ax[2].plot(wl,1-(LRT[:,1]+LRT[:,2]),lw=3,ls='--',c='#9b9b9b')
        ax[2].set_xlim(wl[0],wl[-1])
        ax[2].set_xticks(np.linspace(wl[0],wl[-1],5))
        ax[2].set_xticklabels(np.linspace(wl[0],wl[-1],5),fontsize=26)
        ax[2].set_xlabel('Wavelength (nm)',fontsize=28)
        ax[2].set_ylim(0,1)
        ax[2].set_yticks([0,.25,.5,.75,1])
        ax[2].set_yticklabels([0,.25,.5,.75,1],fontsize=26)
        ax[2].set_ylabel('Absorbance',fontsize=28)
        ax[2].tick_params('both', length=10, width=1.25, which='major',labelsize=24)
        ax[2].plot(wl,1-(LRT[:,1]+LRT[:,2]),lw=3,ls='-',c='black')  
        for axis in ['top','bottom','left','right']:
           ax[2].spines[axis].set_linewidth(1.25)
           
        plt.tight_layout()
        fig.savefig(savestr,dpi=100)
        plt.close()
    
    return opt_params,LRT 

def Prospect5D(wl,N,Cab,Car,Ant,Brown,Cw,Cm):     
    '''
    Prospect5D model to simulate leaf reflectance, transmittance, and
    absorbance from 400 nm - 2500 nm based on foliar and structural 
    properties.
    
    For more details, see the following publication:
    
    Feret J.B., Gitelson A.A., Noble S.D., Jacquemoud S. 2017. PROSPECT5-D:
    Towards modeling leaf optical properties through a complete lifecyle.
    Remote Sensing of Environment 193:204-215
    https ://doi.org/10.1016/j.rse.2017.03.004
        
    Parameters
    ----------
    WL : array_like
        Array of wavelengths. Valid wavelengths range from 400 nm - 2500 nm.
    N : float
        Structure coefficient.
    Cab : float
        Chlorophyll content ug cm-2.
    Car : float
        Carotenoid content ug cm-2.
    Ant : float
        Anthocyanin content ug cm-2.
    Brown : float
        Brown pigment content (arbitrary units).
    Cw : float
        Equivalent water thickness (EWT, cm).
    Cm : float
        Leaf mass per unit area (LMA, g cm-2).

    Returns
    -------
    LRT : array_like
        Array containing WL, modeled reflectance, and modeled transmittance.

    Examples
    --------
    >>> WL  = np.arange(400,2501,1)
    >>> LRT = Prospect5D(WL,1.8,50,10,1,1,0.015,0.008)
    '''
    
    wl_index = wl-400 #Index offset for wavelengths
    
    #WL,nr,Kab,Kcar,Kant,KBrown,Kw,Km
    cfs    = PDB_absorption_coefficients()
    Kall   = (Cab*cfs[wl_index,2]+Car*cfs[wl_index,3]+Ant*cfs[wl_index,4]+
              Brown*cfs[wl_index,5]+Cw*cfs[wl_index,6]+Cm*cfs[wl_index,7])/N;
    j      = np.where(Kall>0)
    
    t1     = (1-Kall)*np.exp(-Kall)
    t2     = (Kall**2)*(sc.exp1(Kall));
    tau    = np.ones(len(t1))
    tau[j] = t1[j]+t2[j]
    
    #Reflectivity and transmissivity at the interface
    talf = calctav(40,cfs[wl_index,1])
    ralf = 1-talf
    t12  = calctav(90,cfs[wl_index,1])
    r12  = 1-t12
    t21  = t12/(cfs[wl_index,1]**2)
    r21  = 1-t21
    
    #Top surface side
    denom = 1-r21*r21*tau**2
    Ta    = talf*tau*t21/denom
    Ra    = ralf+r21*tau*Ta
    
    #Bottom surface side
    t = t12*tau*t21/denom
    r = r12+r21*tau*t
    
    #Reflectance and transmittance of N layers
    D  = np.sqrt((1+r+t)*(1+r-t)*(1-r+t)*(1-r-t))
    rq = r**2
    tq = t**2
    a  = (1+rq-tq+D)/(2*r)
    b  = (1-rq+tq+D)/(2*t)
    
    bNm1    = b**(N-1)        
    bN2     = bNm1**2
    a2      = a**2
    denom   = a2*bN2-1
    Rsub    = a*(bN2-1)/denom
    Tsub    = bNm1*(a2-1)/denom
    
    #Case of zero absorption
    j       = np.where(r+t >= 1);
    Tsub[j] = t[j]/(t[j]+(1-t[j])*(N-1));
    Rsub[j]	= 1-Tsub[j];
    
    #Reflectance and transmittance of the leaf: combine top layer with next N-1 layers
    denom    = 1-Rsub*r
    trans    = Ta*Tsub/denom
    refl     = Ra+Ta*Rsub*t/denom
    
    return np.column_stack((wl,refl,trans))