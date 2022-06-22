

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import pandas as pd
from pandas import DataFrame
import random
import glob
import os
from scipy import interpolate


def IMF_Kroupa(masses, alpha):
    IMF=np.array([0 for i in masses],dtype=float)
    dy1=(np.power(.5,-alpha[2])-np.power(.5,-alpha[1]))
    dy2=dy1+(np.power(.08,-alpha[1])-np.power(.08,-alpha[0]))
    for i,mass in enumerate(masses):
        #print(i,mass,alpha[1],IMF[i],np.power(mass,-alpha[0]))
        if         mass <= .08 : IMF[i]=np.power(mass,-alpha[0])+dy2
        elif .08 < mass <= .5  : IMF[i]=np.power(mass,-alpha[1])+dy1
        else                   : IMF[i]=np.power(mass,-alpha[2])
    #print(np.power(.5,-alpha[2]) )       
    return IMF

def SSP_generator(isochrone, bands, n_stars, n_mass_grid=10000000 ): 
    # Input isochrone should be a dataframe.
    # Input n_star is the number of generated stars.
    mass_min = np.min(isochrone[1]['Mini'])
    mass_max= np.max(isochrone[1]['Mini'])
    mass_bin=np.linspace(mass_min,mass_max, n_mass_grid) # creat the mass grid
    
    IMF=IMF_Kroupa(mass_bin,[.3,1.3,2.3]) 
    new_masses=random.choices(mass_bin, weights=IMF, k=n_stars) # generate new masses
    SSP = pd.DataFrame(new_masses , columns=['Mini'])           # make a df for the SSP

    logAge = isochrone[0][0]
    SSP['logAge']= np.full(n_stars,logAge)
    
    MH = isochrone[0][1]
    SSP['MH']= np.full(n_stars,MH)
    
    for band in bands:
        x = isochrone[1]['Mini']
        y = isochrone[1][band]
        f = interpolate.interp1d(x, y)
        #globals()[' %s' % band] = f(new_masses) 
        SSP[band]= f(new_masses)  # estimate the mags for the generated masses and add to the SSP
    return SSP

    isochrones={}


def add_uncerntaties(magnitudes,fit_param, mag_feature, SN_feature):
    
    # INPUT: 
    
    # magnitudes: magnitudes of simulated sources
    # fit param: best fit parameters found by fiting delra_m=exp(a+b*m) where m and delta_m are the magnitudes and magnitude uncertanties of the reference data for uncertainties!
    # mag_feature: This is the apparant magnitude of your desired feature which you want to have the S/N ratio of SN_feature
    # SN_feature: SN_feature for mag_feature
    
    #######################################
    
    # OUTPUT: 
    
    # mag_with_unc: Input magnitudes + randon gaussian uncertainties
    # delta_m_final: Sigma of random gaussian uncertainties
    # exp_time_final: The ratio of needed exposure time to the exposure time of the reference data to get to SN_feature for mag_feature
    
    Xn = np.c_[np.ones(magnitudes.size), magnitudes]
    delta_m_initial = poisonModel_Prediction_unc(fit_param, Xn)
    delta_m_feature_initial = poisonModel_Prediction_unc(fit_param, np.c_[np.ones(1), mag_feature])
    
    SN_feature_initial = 1/(10**(-.4*(-delta_m_feature_initial))-1)
    exp_time_initial = 1
    exp_time_final = exp_time_initial*(SN_feature/SN_feature_initial)**2
    delta_m_final = np.abs(-2.5*np.log10(1 + ( (exp_time_initial/exp_time_final)**.5*(10**(-.4*-delta_m_initial)-1) ) ))
    mag_with_unc = np.random.normal(magnitudes, delta_m_final, magnitudes.shape[0])
    return mag_with_unc,delta_m_final,exp_time_final

def poisonModel_Prediction_unc(b,X):
    yhat = np.exp(X @ b)
    return yhat

###############################
###############################
###############################

isochrones={}
AGE=[]

METALLICITY=[]
isoch_files=['iso0.txt']
for file in isoch_files:
    parsec=[]
    for txt in [file]:
        mylines = []                             # Declare an empty list named mylines.
        with open (txt, 'rt') as myfile: # Open lorem.txt for reading text data.
            for myline in myfile:                # For each line, stored as myline,
                mylines.append(myline)           # add its contents to mylines.  
        #print(len(mylines))
        for i in range(len(mylines)):
            if mylines[i][0] == '#':
              continue
            isoch=mylines[i].split()
            for j in range(len(isoch)):
                isoch[j]=float(isoch[j])
            parsec.append(isoch) 
    #print(mylines[13].split()[1:])
    stellarpop=pd.DataFrame(parsec,columns=mylines[13].split()[1:])
    stellarpop=stellarpop[stellarpop['label']!=9]  # drop Post AGB points because those mess with star generation!
    logage=np.sort(stellarpop['logAge'].value_counts().index.tolist())
    MHs = np.sort(stellarpop['MH'].value_counts().index.tolist())
    print(logage,MHs)
    AGE.append(logage)
    METALLICITY.append(MHs)
    for age in logage:
        for metallicity in MHs:
            isochrones[f'logAge:{age:.2f}_MH:{metallicity:.2f}']=[[age,metallicity],stellarpop[(stellarpop['logAge']==age) & (stellarpop['MH']==metallicity)]]

parsec=[]
isochrone_names=list(isochrones.keys())

###############################
###############################
###############################

bands=['F606Wmag','F814Wmag']
SSP = SSP_generator(isochrones[isochrone_names[0]], bands, n_stars=100000)
print(SSP.head())
SSP.head()

###############################
###############################
###############################
abs_mag_feature = np.array([5.5, 4.5]) 

distandmodulus = 24
print('distandmodulus= ', distandmodulus)
mag_feature = abs_mag_feature + distandmodulus
SN_feature = [10,10]
print('mag_feature=', mag_feature)

bands_unc_coef=[[-23.85588548 , 0.7982657 ], [-24.53558396 , 0.84130151]]  #IC1613

colour_min = -1
colour_max = 4

magnitude_max = mag_feature
magnitude_min = magnitude_max - 10


aa = SSP[bands[0]] + distandmodulus
bb = SSP[bands[1]] + distandmodulus

aa = aa[( bb < magnitude_max[1]+1.193 )]
bb = bb[( bb < magnitude_max[1]+1.193 )]

print(bb.shape,bb.shape)

aa = add_uncerntaties(aa,fit_param=bands_unc_coef[0], mag_feature = mag_feature[0], SN_feature = SN_feature[0])[0]
bb = add_uncerntaties(bb,fit_param=bands_unc_coef[1], mag_feature = mag_feature[1], SN_feature = SN_feature[1])[0]


aa = aa[( bb < magnitude_max[1]+1.193 )]
bb = bb[( bb < magnitude_max[1]+1.193 )]

x = aa-bb
y = bb

#plt.scatter(x,y,s=10,c=SSP['Mini'])
plt.scatter(x,y,s=1)
#plt.scatter(stellarpop.F555Wmag-stellarpop.F814Wmag,stellarpop.F814Wmag,c=stellarpop.Mass,s=5, cmap='cool') #,label='logAGE=%.2f'%(logAGE)
plt.gca().invert_yaxis()
plt.colorbar(label='mass')
plt.xlabel('g-i')  #don't forget chanfe the axis title if you chose different bands!
pl.rcParams["figure.figsize"] = (8,8)
#plt.xlim(colour_min,2)
print(magnitude_max[1],magnitude_min[0])
#plt.ylim(magnitude_max[1],magnitude_min[0])
plt.ylabel('g')
plt.legend()
plt.show()


