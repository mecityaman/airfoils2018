# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 13:12:05 2018
Finished Mon Jan 22 12:30

@author: myaman
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

import foil

## this script was used to create newbase pandas
## to hold pointed/cusped parameter information.
## input dataframe: rawbase.pkl
## output dataframe:rawbase_with_structure_info.pkl


FOIL_DATABASE_FILE = 'data/rawbase.pkl'
baseframe           = foil.get_foils_frame(FOIL_DATABASE_FILE)

n_          = 10 # number of points used for linear regression
threshold   = 5.6 

l__             = []
dlist           = []
deltathetalist  = []

for i,j in baseframe.iterrows():
    
    n = i
    
    x__, y__    = baseframe['x_nn'][n],baseframe['y_nn'][n]
    
    xup, yup    = x__[:n_], y__[:n_]
    xlo, ylo    = x__[-n_:],y__[-n_:]
    
    regup       = stats.linregress(xup,yup)
    reglo       = stats.linregress(xlo,ylo)
    
    xx          = np.linspace(.8,1.2,50)
    yyup        = regup[0]*xx+regup[1]
    yylo        = reglo[0]*xx+reglo[1]
    theta1      = np.arctan2(regup[0],1)
    theta2      = np.arctan2(reglo[0],1)
    delta       = -(theta1-theta2)/np.pi*180
    
    
    print('%5.2f %5.2f %5.2f'%(theta1, theta2, theta1-theta2))
        
    divergence = 100*(yylo[-1]- yyup[-1])
    
    dlist.append(divergence)
    deltathetalist.append(delta)


baseframe['div'] = dlist
baseframe['delta_theta'] = deltathetalist

baseframe=baseframe.sort_values('div')
baseframe['is_directed'] = baseframe['div']>threshold
#plt.plot(baseframe['div'].values)

plt.title('Trailing edge angle (sorted)')
plt.text(400,30,'Cusped')
plt.text(1080,30,'Directed')
plt.xlabel('Airfoil')
plt.ylabel('Angle')
plt.plot(baseframe['delta_theta'].values)
plt.plot(52*baseframe['is_directed'].values)
plt.show()

if False:
  baseframe.to_pickle('data/rawbase_with_structure_info.pkl')

    
    