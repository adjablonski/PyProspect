import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import os
from scipy.interpolate import interp1d
os.chdir('C:/Users/Andrew/Documents/GitHub/PyProspect/')
from PyPROSPECT5d import Prospect5D,inversion5D_reflectance,inversion5D_reflectance_transmittance

###Simulate spectra using Prospect5D model###

N      = 1.8   #structure coefficient
Cab    = 50.0  #chlorophyll content (µg.cm-2) 
Car    = 10.0  #carotenoid content (µg.cm-2)
Ant    = 1.0   #Anthocyanin content (µg.cm-2)
Brown  = 0.1   #brown pigment content (arbitrary units)
Cw     = 0.015 #EWT (cm)
Cm     = 0.008 #LMA (g.cm-2)

#Prospect model starts at 400 nm
wl  = np.arange(400,2501,1)
LRT = Prospect5D(wl,N,Cab,Car,Ant,Brown,Cw,Cm)

###Example of model inversion with reflectance data only###
spec = pd.read_csv('C:/Users/Andrew/Dropbox/PERS_Lab/UAV_MissionData/TraitData/20200923/20200923_SpectraLMA.csv')
wl   = np.arange(400,801,1)
refl = spec.values[1,5:]
refl = refl[50:801-350]

params  = {'N':None,'Cab':None,'Car':None,'Ant':None,'Brown':None,'Cw':None,'Cm':spec.values[1,3]}
savestr = 'C:/Users/Andrew/Dropbox/PERS_Lab/UAV_MissionData/TraitData/20200923/Prospect5D_Output/F1L2.png'

coefs,LRT = inversion5D_reflectance(wl,refl,params,savestr)


###Example of model inversion with reflectance and transmittance data###

params  = {'N':None,'Cab':None,'Car':None,'Ant':None,'Brown':None,'Cw':None,'Cm':0.007211}
savestrA = 'C:/Users/Andrew/Documents/GitHub/PyProspect/ExampleOutput/Example2_Leaf1_RTA_abs.png'
savestrB = 'C:/Users/Andrew/Documents/GitHub/PyProspect/ExampleOutput/Example2_Leaf1_relectance.png'

wl    = np.loadtxt('C:/Users/Andrew/Documents/GitHub/PyProspect/ExampleData/Example2_wl.csv',delimiter=',')
refl  = np.loadtxt('C:/Users/Andrew/Documents/GitHub/PyProspect/ExampleData/Example2_Reflectance.csv',delimiter=',')
trans = np.loadtxt('C:/Users/Andrew/Documents/GitHub/PyProspect/ExampleData/Example2_Transmittance.csv',delimiter=',')

refl_int = interp1d(wl,refl)
refl_int = refl_int(np.arange(400,801,1,))

trans_int = interp1d(wl,trans)
trans_int = trans_int(np.arange(400,801,1,))

wl = np.arange(400,801,1)

coefs,LRT   = inversion5D_reflectance_transmittance(wl,refl_int[0,:],trans_int[0,:],params,savestrA,plot=True)
coefsB,LRTB = inversion5D_reflectance(wl,refl_int[0,:],params,trans_int[0,:],savestrB,plot=True)

np.sqrt(np.mean(((1-(LRTB[:,1]+LRTB[:,2]))-(1-(refl_int[0,:]+trans_int[0,:])))**2))
np.sqrt(np.mean((LRTB[:,2]-trans_int[0,:])**2))