# -*- coding: utf-8 -*-
"""
author  : myaman@thk.edu.tr
created : Wed Oct 11 13:59:11 2017
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_foils_frame(datafile):
 
    return pd.read_pickle(datafile)

def get_a_foil(pandasframe, i):
        
    x    = pandasframe.get_value(i,'x_nn')
    y    = pandasframe.get_value(i,'y_nn')
    
    return x, y

def get_foil_info(pandasframe,i):
    
    name            = pandasframe.get_value(i,'name').split('--')[0].strip()
    n_of_panels     = len(pandasframe.get_value(i,'x_nn'))
    summary         = {'index':i,
                       'name':name,
                       'n_of_panels':n_of_panels,
                       'directed':bool(pandasframe.get_value(i,'is_directed'))
                       #,'camber':False
                       #,'chord' :False
                       }
    return summary

def panel_parameters(afoil, is_directed=''):
      
    x, y = afoil
            
    # panel centers
    xp = (1/2)*(x[1:]+x[:-1])
    yp = (1/2)*(y[1:]+y[:-1])
    
    # panel lengths
    p_l = np.sqrt((x[1:]-x[:-1])**2 +(y[1:]-y[:-1])**2)
    
    # panel tangents(tx,ty) as unit vectors
    tx = x[1:]-x[:-1]
    ty = y[1:]-y[:-1]
    n_ = np.linalg.norm(np.vstack((tx,ty)), axis=0)

    p_tx = tx/n_
    p_ty = ty/n_

    # panel normals (nx,ny) as unit vectors
    p_nx =  p_ty[:]
    p_ny = -p_tx[:]
    
    # measurement points can be located differently
    # do not include trailing edge when enforcing zero normal velocity condition
    meas_x      = x[1:-1]
    meas_y      = y[1:-1]
    
    # unit normals/tangents at meas_x and meas_y
    # each of these points are in between two panel centers
    # so interpolate panel normals/ tangents to get
    # normals and tangents at the measurement points
        
    fnx         = (p_nx[1:]+p_nx[:-1])/2
    fny         = (p_ny[1:]+p_ny[:-1])/2
    n_          = np.linalg.norm(np.vstack((fnx,fny)), axis=0)
    meas_nx     = fnx/n_
    meas_ny     = fny/n_

    ftx     = (p_tx[1:]+p_tx[:-1])/2
    fty     = (p_ty[1:]+p_ty[:-1])/2
    n_      = np.linalg.norm(np.vstack((ftx,fty)), axis=0)
    meas_tx     = ftx/n_
    meas_ty     = fty/n_
        
    # to enforce Kutta condition for both the directed and cusped foils,
    # ie the trailing edge must be a stationary point for the pointed, and 
    # the upper and lower surfaces must have equal limiting velocities.
    # remove a random measurement point, here we
    # choose to remove the 5th measurement point. 
    # equations are set in the vortex solver.
    # below we repeat the same for each case, actually the if statements
    # should be merged.
    
    if is_directed == 1.0:
    
        ele         = 5
        meas_x      = np.delete(meas_x,ele)
        meas_y      = np.delete(meas_y,ele)
    
        meas_nx      = np.delete(meas_nx,ele)
        meas_ny      = np.delete(meas_ny,ele)
        
        meas_tx      = np.delete(meas_tx,ele)
        meas_ty     = np.delete(meas_ty,ele)
   
    
    if is_directed == 0.0:
       
        #same as above
        ele         = 5
        meas_x      = np.delete(meas_x,ele)
        meas_y      = np.delete(meas_y,ele)
    
        meas_nx      = np.delete(meas_nx,ele)
        meas_ny      = np.delete(meas_ny,ele)
        
        meas_tx      = np.delete(meas_tx,ele)
        meas_ty     = np.delete(meas_ty,ele)
        
        
        
    panel_params = xp, yp, p_l, p_nx, p_ny, p_tx, p_ty
    meas_params  = meas_x, meas_y, meas_nx, meas_ny, meas_tx, meas_ty
 
    return panel_params, meas_params

       
if __name__ == '__main__':
   
    
    FOIL_DATABASE_FILE          = 'data/rawbase.pkl'
    #FOIL_DATABASE_FILE          = 'testfoils.pkl'
    foilsframe                  = get_foils_frame(FOIL_DATABASE_FILE)
    
    for i, row in foilsframe.iterrows():
        if foilsframe.get_value(i,'use'):
            is_directed = foilsframe.get_value(i,'is_directed')
            
            print(i)
            x_foil, y_foil              = afoil = get_a_foil(foilsframe,i)    
            panel_params, meas_params   = panel_parameters(afoil, is_directed=is_directed)
                        
            print('len x_foil',len(x_foil))
            xp, yp, p_l, p_nx, p_ny, p_tx, p_ty                 = panel_params
            meas_x, meas_y, meas_nx, meas_ny, meas_tx, meas_ty  = meas_params

            
            print()                        
            print('len x_foil',len(x_foil))
            print('len xp',len(xp))
            
            
            print()                        
            print('len meas_x',len(meas_x))
            
            
            plt.rcParams['figure.figsize'] = [16.0, 16.0]
            plt.axes().set_aspect(1)
            
            plt.plot([-.10,1.10],[-0.20,0.20],'.', alpha=0.1)
            plt.plot(x_foil, y_foil,'g.-', alpha=0.25)
            titlestring = [i for i in get_foil_info(foilsframe,i).values()]
            plt.title(titlestring)
                    
            #plt.quiver(xp, yp, p_nx, p_ny, alpha=.1)
            #plt.quiver(xp, yp, p_tx, p_ty, alpha=.1)
            
            plt.quiver(meas_x,meas_y,meas_nx,meas_ny, alpha=0.05, color='b')
            #plt.quiver(meas_x,meas_y,meas_tx,meas_ty, alpha=.5, color='b')
            
            #plt.axes([0.78, 0.56, 0.1, 0.1], facecolor='y')
            #plt.plot(np.abs(np.arctan(meas_ny,meas_nx)*180.0/np.pi))
                
            plt.show()
            time.sleep(5)
