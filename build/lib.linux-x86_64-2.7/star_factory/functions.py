import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import pandas as pd
import random
from scipy import interpolate

#####
def IMF_Kroupa(masses, alpha):

    """Initial mass function

    Calculate the Kroupa IMF.

    Args:
        masses (array): numpy vector. An array of mass between the min and max masses.
        alpha (list): A list with three elements which are the Kroupa IMF slopes.

    Returns:
        array: An array
    """


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
