#-*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:11:58 2017

@author: myaman
"""

import os, time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal

np.set_printoptions(precision=2, suppress=True)


def convert_text_file_database_to_pandas_frame(folder = 'coord_seligFormat'):
    
    foil_dict   = {}
    
    for i, filename in enumerate(os.listdir(folder)):
    
        with open(folder+'/'+filename, 'r',  encoding='ANSI') as f:
            head = f.readline()
            foilname = head[:-1].strip()
        
        print(i, filename,' ', foilname)
        
        try:
            foil    = pd.read_csv(folder+'/'+filename, skiprows=1,
                                  engine='python', header=None, delim_whitespace=True)
            plt.plot(foil[0],foil[1], alpha=.05)
        except:
            print('encountered problem with file ,',i, filename)
            time.sleep(10)
              
        foil_dict[i]={'file name': filename,
                     'name': foilname,
                     'x': foil[0].values,
                     'y': foil[1].values}
                     
    frame = pd.DataFrame(foil_dict).transpose()
    print(frame.head())
    
    return frame
    

def resample_foil_panels(frame, n_of_panels):

    aframe = frame.copy(deep=True)
    
    for i,j in aframe.iterrows():
        
        x, y = j['x'],j['y']
        fx = signal.resample(x,n_of_panels)
        fy = signal.resample(y,n_of_panels)
        
        aframe.set_value(i,'x', fx)
        aframe.set_value(i,'y', fy)
                
    
    new_file_name =  'panels-'+str(n_of_panels)+'.pkl'
    aframe.to_pickle(new_file_name)
    print('\n', new_file_name, ' written.')

    return aframe
    
    
def analyze_foils(frame):
    
    '''
    'leading edge'  : [x,y]
    'trailing edge' : [x,y] 
    'closed'        : True/False,
    'cusped'        : True/False,
    'chord line'    : [[],[]]    # locus of points from the trailing edge to the leading edge.
    'camber line'   : [[],[]]  # locus of points halfway between the upper and lower surfaces
       
     'thickness'    : []
     'camber':      : f   # maximum distance between the chordline and camber line
    '''
                 
    aframe = frame.copy(deep=True)
    adict = dict(aframe.transpose())
    
    count = 0
    
    for i,j in adict.items():
        
        x = j['x']
        y = j['y']
        name = j['file name']
        
        condition = ((np.abs(x[0]-1.0) > 0.01) or
                 (np.abs(x[-1]-1.0) > 0.01) or
                 (np.abs(y[0]-0.0) > 0.01) or
                 (np.abs(y[-1]-0.0) > 0.01))
        
        
        if condition:
            print(i, name, x[:5],x[-5:])
            plt.title(name)
            plt.plot(x,y)
            #plt.show()
            count += 1
   
    print()
    print(count, 'found.')
    
    newframe = pd.DataFrame(adict).transpose()
    
    return newframe
    
    

if __name__ == '__main__':

    ''''
    reloads foildatabase.pkl or makes up a pandasframe from the text files
    '''
    
    if os.path.exists('foildatabase.pkl'):
        frame = pd.read_pickle('foildatabase.pkl')
    else:
        frame = convert_text_file_database_to_pandas_frame()
        frame.to_pickle('foildatabase.pkl')
           
    '''
    writes a new frame with specified sampling rate
    '''
    newframe = resample_foil_panels(frame, 100)
    
    for i in range(len(newframe)):
        
        plt.plot(newframe.get_value(i,'x'),newframe.get_value(i,'y'), alpha=.1)
