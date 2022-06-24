import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import pandas as pd
import random
import functions as func

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
    #print(logage,MHs)
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
SSP = func.SSP_generator(isochrones[isochrone_names[0]], bands, n_stars=100000)
print(SSP.shape)

###############################
###############################
###############################
abs_mag_feature = np.array([5.5, 4.5]) 

distandmodulus = 24
#print('distandmodulus= ', distandmodulus)
mag_feature = abs_mag_feature + distandmodulus
SN_feature = [10,10]
#print('mag_feature=', mag_feature)

bands_unc_coef=[[-23.85588548 , 0.7982657 ], [-24.53558396 , 0.84130151]]  #IC1613

colour_min = -1
colour_max = 4

magnitude_max = mag_feature
magnitude_min = magnitude_max - 10


aa = SSP[bands[0]] + distandmodulus
bb = SSP[bands[1]] + distandmodulus

aa = aa[( bb < magnitude_max[1]+1.193 )]
bb = bb[( bb < magnitude_max[1]+1.193 )]


aa = func.add_uncerntaties(aa,fit_param=bands_unc_coef[0], mag_feature = mag_feature[0], SN_feature = SN_feature[0])[0]
bb = func.add_uncerntaties(bb,fit_param=bands_unc_coef[1], mag_feature = mag_feature[1], SN_feature = SN_feature[1])[0]


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
#print(magnitude_max[1],magnitude_min[0])
#plt.ylim(magnitude_max[1],magnitude_min[0])
plt.ylabel('g')
plt.legend()
plt.show()


