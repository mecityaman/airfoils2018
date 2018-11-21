# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 11:43:45 2017

@author: myaman@thk.edu.tr
"""

import sympy
from sympy.abc import x, y
import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd

import foil
#import panel_parameters
import flow_functions
import time

'''
A vortex is placed at each panel center. Then flow at measurement points 
due to each vortex is calculated using the vortex flow function. Resultant 
velocity vector at each measurement point  due to vortex sources is simply 
the superposition of the flow due to each vortex. 

Next, the boundary condition at each measurement point, i.e. the normal 
component of the veleocity vector must be zero, is enforced. We do this using 
a square matrix. Each row, an independent linear equation, is a measurement 
point. The number of vortices is two more than the measurement points.
Two additional equations are obtained from the Kutta condition, i.e. 
the velocity at a (a) pointed trailing edge must be zero (b) and for a cusped
trailing edge the upper and lower panel limit must be the same. 

[Vortex matrix] [vortex intensity] = [-Uniform flow velcoity]

V * i = U -> i = inv(V) * U


get vortex square matrix and free flow column matrix
vortex square matrix is (# of meas points + 2 kutta lines, # of vortices)


'''

def get_free_flow_normals(free_flow = (1.0,0.0), geometry_params = ''):

    # returns normal components of the free flow velocity at measurement pts.
    # and fills two zeros at the end to make a column matrix of (# of vortices,1)
    
    panel_params, meas_params                           = geometry_params
    xp, yp, p_l, p_nx, p_ny, p_tx, p_ty                 = panel_params
    meas_x, meas_y, meas_nx, meas_ny, meas_tx, meas_ty  = meas_params

    fsvx, fsvy                                          = free_flow

    free_stream_normal_components                       = (fsvx * meas_nx +
                                                           fsvy * meas_ny)
    
    vel_matrix = np.hstack((free_stream_normal_components, np.zeros(2)))
        
    return vel_matrix


def get_vortice_matrice(afoil='', vortex_intensity = '', geometry_params = '', is_directed=''):

    # returns a matrix, when multiplied with vortex intensities gives the 
    # resultant (normal) velocities at measurement points due to all vortices.
    
    x_foil, y_foil = afoil
    
    panel_params, meas_params                           = geometry_params
    xp, yp, p_l, p_nx, p_ny, p_tx, p_ty                 = panel_params
    meas_x, meas_y, meas_nx, meas_ny, meas_tx, meas_ty  = meas_params
   
    
    # put a vortex at each panel center
    n__                 = len(xp)
    n_                  = list(range(n__)) 
    
    # the position of the vortex
    vortex_x, vortex_y  = xp[n_], yp[n_]
    
    # vortices are scaled with panel length.
    vortex_l            = p_l[n_]
    
    
    # if no vortex_intensity is provided we are calculating the vortex matrix
    # else we use a solution vortex vector to verify computation.
    
    if vortex_intensity == '':
        vortex_i = np.ones((n__)) * 1.0
        #print('taking default 1.0 values')
    else:
        vortex_i = vortex_intensity
        #print('inserting vortex values')

    # zip vortex information        
    vortices    = list(zip(vortex_x, vortex_y, vortex_i, vortex_l, n_))
    
    # placeholder for vortex matrix V
    # equations for measurements points
    mp          = np.zeros((n__-2, n__))
    
    # equations for Kutta condition
    mp_appendix = np.zeros((2,n__))
    
    
    for x_, y_, t_, l_, i_ in vortices:
   
        # generate function for a vortex
        u,v  = flow_functions.vortex_velocities_lambda(flow_functions.vortex(x_, y_,t_,l_))
        # plug-in measurement points to calculate numerical u,v values
        # and get normal components by the scalar product with meas_nx, meas_ny
        dotp = u(meas_x,meas_y) * meas_nx + v(meas_x,meas_y) * meas_ny
        

        mp[:,i_]            = dotp
        # matrix eqn will hold only if u(1.0,0.0) and v(1.0,0.0) are zero.
    
        if is_directed=='':
            print('there"s a problem.')
            print('\a')
            time.sleep(100)
        
        
        if is_directed == 1.0:
            
            mp_appendix[-2,i_]  = u(1.0,0.0)
            mp_appendix[-1,i_]  = v(1.0,0.0)
        
        if is_directed == 0.0:
            
            #alternatively one can try a nearer point, but this should suffice.
            
            upper_point_x, upper_point_y = (meas_x[0]+ x_foil[0])/2,(meas_y[0]+ y_foil[0])/2
            lower_point_x, lower_point_y = (meas_x[-1]+ x_foil[-1])/2,(meas_y[-1]+ y_foil[-1])/2

            '''
            plt.plot(x_foil[:5],y_foil[:5])
            plt.plot(x_foil[-5:],y_foil[-5:])
            
            plt.plot(meas_x[:5],meas_y[:5],'.', alpha=.2)
            plt.plot(meas_x[-5:],meas_y[-5:],'.', alpha=.2)
            plt.plot(upper_point_x,upper_point_y,'b.')
            plt.plot(lower_point_x,lower_point_y,'b.')

            plt.show()            
            '''
            # both components are forced to be equal at the upper and lower points
            mp_appendix[-2,i_]  = u(upper_point_x,upper_point_y) - u(lower_point_x,lower_point_y)
            mp_appendix[-1,i_]  = v(upper_point_x,upper_point_y) - v(lower_point_x,lower_point_y)
            
        

    vortices_matrix = np.vstack((mp,mp_appendix))
    
    return  vortices_matrix


def solve_for_vortex_intensity_vector(afoil='',
                                      flow_speed='',
                                      is_directed=''):
    
    
    panel_params, meas_params = params                  = foil.panel_parameters(afoil, is_directed=is_directed)
    xp, yp, p_l, p_nx, p_ny, p_tx, p_ty                 = panel_params
    meas_x, meas_y, meas_nx, meas_ny, meas_tx, meas_ty  = meas_params
    
    #set flow speed
    flow_speed = flow_speed
    
    '''
    if 0:
        #draw panel geometry    
        plt.rcParams['figure.figsize'] = [16.0, 16.0]
        title_string = foilname+' ' +str(len(afoil[0]))+(' points, analysis with '
                                        +str(len(xp))+' vortices and '+ str(len(meas_x))
                                        +' measurement points')
        plt.title(title_string)
        plt.axes().set_aspect(1)
        plt.plot([-.15,1.15],[-0.15,0.15],'.', alpha=0.1)
        plt.plot(afoil[0], afoil[1],'-o')
        plt.quiver(xp, yp, p_nx, p_ny, alpha=.1)
        #plt.quiver(xp, yp, p_tx, p_ty, alpha=.1)
        #plt.quiver(meas_x,meas_y,meas_nx,meas_ny, alpha=.5, color='b')
        #plt.quiver(meas_x,meas_y,meas_tx,meas_ty, alpha=.5, color='b')
    '''
    
    # calculate free flow normals using panel geometry
    vel = get_free_flow_normals(free_flow=flow_speed, geometry_params=params)
    
    '''
    if 0:
        #draw geometry
        plt.quiver(meas_x, meas_y, vel[:-2]*meas_nx, vel[:-2]*meas_ny, alpha=0.1, color='b', scale=.5, scale_units='inches')
        vvv=np.sum(vor,axis=1)[:-2]
        plt.quiver(meas_x, meas_y, vvv*meas_nx ,vvv*meas_ny, color='r', alpha=.35, scale=.5, scale_units='inches')
        plt.quiver(meas_x, meas_y, vvv*meas_nx +vel[:-2]*meas_nx ,vvv*meas_ny + vel[:-2]*meas_ny, color='r', alpha=.35, scale=.5, scale_units='inches')
    '''
    
    # calculate vortice factors matrice
    
    vor = get_vortice_matrice(afoil=afoil, geometry_params=params, vortex_intensity='', is_directed=is_directed)
    #print(vor)
    
    vsol = np.matmul(np.linalg.inv(vor),-vel)
    '''    
    if 0:
        #verify solution
        n__                 = len(xp)
        n_                  = list(range(n__)) # put a vertex at each panel center
        vortex_x, vortex_y  = xp[n_], yp[n_]
        vortex_l            = p_l[n_]
        #vortex_i            = np.ones_like(vsol)
        vortex_i            = vsol
        
        vortices            = list(zip(vortex_x, vortex_y, vortex_i,vortex_l,n_))
        
        u,v = flow_functions.vortices(vortices, free_flow=flow_speed)
        
        Y, X    =  np.ogrid[-.3:.3:500j, -.2:1.2:500j] 
        speed   = np.sqrt(u(X,Y)**2 + v(X,Y)**2)
        lw      = 50*speed/ speed.max()
        strm    = plt.streamplot(X, Y, u(X, Y), v(X, Y), 
                            color='cornflowerblue', linewidth=speed)

    '''
    
    return vsol

    
    

if __name__ == '__main__':
  
  
    

    FOIL_DATABASE_FILE          ='data/rawbase_with_structure_info.pkl'
    
    foilsframe                  = foil.get_foils_frame(FOIL_DATABASE_FILE)
    
    foilsframe['vortices']      = foilsframe['x_nn']*0.0 #creating space for vsols
    foilsframe['lift']          = np.NaN 
    flow_speed                  = (1.0, 0.0) 
    
    counter = 0
    for i, row in foilsframe.iterrows():
        if row['use']:
            if 1.0:
                counter=counter+1
                
                is_directed = row['is_directed']
                                               
                x_foil, y_foil              = afoil = foil.get_a_foil(foilsframe, i)

                panel_params, meas_params   = foil.panel_parameters(afoil, is_directed=is_directed)
                
                xp, yp, p_l, p_nx, p_ny, p_tx, p_ty                 = panel_params
                meas_x, meas_y, meas_nx, meas_ny, meas_tx, meas_ty  = meas_params
           
                vsol = solve_for_vortex_intensity_vector(afoil = afoil,
                                                         flow_speed = flow_speed,
                                                         is_directed = is_directed)
               
                text = '#%4d foil %04d, lift %5.2f,' %(counter, i,np.sum(vsol))
                if is_directed:
                    text += 'pointed (%5.2f)'%row['div']
                else:
                    text +=' cusped (%5.2f)'%row['div']
                print(text)
                
                foilsframe.set_value(i,'vortices',vsol)
                foilsframe.set_value(i,'lift', np.sum(vsol))
               
                if True:    
                
                    plt.rcParams['figure.figsize'] = [10.0, 10.0]
                    fig=plt.figure()
                    ax5 = fig.add_axes([0.0,0.0,1,1], frameon=True) 
                    ax5.set_aspect('equal')
                    plt.title(text)
                    
                    n__                 = len(xp)
                    n_                  = list(range(n__)) # put a vertex at each panel center
                    vortex_x, vortex_y  = xp[n_], yp[n_]
                    vortex_l            = p_l[n_]
                    #vortex_i           = np.ones_like(vsol)
                    vortex_i            = vsol
                    vortices            = list(zip(vortex_x, vortex_y, vortex_i,vortex_l,n_))
                    
                    u,v = flow_functions.vortices(vortices, free_flow=flow_speed)
                    
                    Y, X    =  np.ogrid[-.2:.2:500j, -.1:1.1:500j] 
                    speed   = np.sqrt(u(X,Y)**2 + v(X,Y)**2)
                    lw      = 50*speed/ speed.max()
                    
                    ax5.plot([-.10,1.10],[-0.15,0.15],'.', alpha=0.1)
                    ax5.plot(xp,yp, alpha=1.0)
                    
                    strm = ax5.streamplot(X, Y, u(X, Y), v(X, Y), 
                                        color='cornflowerblue', linewidth=speed)
                    
                    ax6 = fig.add_axes([0.88, 0.34,0.12,0.12], frameon=False)
                    
                    ax6.matshow(speed[::-1], cmap='gray', vmin=0, vmax=1)
                    
                    plt.xticks([])
                    plt.yticks([])
                    
                    params = '%04d '%counter + '%04d '%i +  '%5.2f' %np.sum(vsol)
                    
                    fname = 'figures/performance/'+text+'.png'
                    
                    plt.savefig(fname)#, dpi=fig.dpi)
                    plt.show()



    if False:

      # these foils fail to produce infite lifts, so drop them.
      foilsframe.drop(235, inplace=True)
      foilsframe.drop(377, inplace=True)
      foilsframe.drop(869, inplace=True)
      
      foilsframe=foilsframe.sort_values('lift')
      foilsframe.to_pickle('data/rawbase_with_structure_and_lift.pkl')                            