# -*- coding: utf-8 -*-
"""
DNNRegressor with custom input_fn.
Created on Wed Oct  4 10:48:44 2017
@author: myaman
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, pickle
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import foil, convert_to_Tensorflow_input_file

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

def draw_neural_net(ax, left, right, bottom, top, layer_sizes, weights1='', weights2='', weights3=''):
    '''
    Draw a neural network cartoon using matplotilb.
    
    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])
    
    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    '''
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    # Nodes
    
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            nodecolor='grey'
            if n==0:
                if m>layer_size/2:
                    nodecolor='r'
                else:
                    nodecolor='b'
                
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='w', ec=nodecolor, zorder=4, alpha=.82)
            
            
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        #print(n, layer_size_a, layer_size_b)
        
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        
        if n==0: 
            lw=.8
        
        else:
            lw=0.2
            
        minima = weights1.min()
        maxima = weights1.max()
        
        norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('jet'))

        for m in range(layer_size_a):
            
            for o in range(layer_size_b):
                if n==0:
                    line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                      [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing],
                                      c=mapper.to_rgba(weights1[m,o]), alpha=.07, linewidth=lw)
                    ax.add_artist(line)
                elif n==1:
                    line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                      [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing],
                                      c=mapper.to_rgba(weights2[m,o]), alpha=.7, linewidth=lw)
                    ax.add_artist(line)
                
                elif n==2:
                    line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                      [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing],
                                      c=mapper.to_rgba(weights3[m,o]), alpha=.7, linewidth=lw)
                    ax.add_artist(line)
    #plt.colorbar(mapper([weights1]))

def get_input_fn(data_set, num_epochs=None, shuffle=True):
    # function to streamline data from pandas into TensorFlow
    return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
      y=pd.Series(data_set[LABEL].values),
      num_epochs=num_epochs,
      shuffle=shuffle)

def get_pandas_file_for_lift():
    FOIL_DATABASE_FILE = 'data/rawbase_with_structure_and_lift.pkl'
    baseframe           = foil.get_foils_frame(FOIL_DATABASE_FILE)
    #baseframe           = baseframe[baseframe['use']==True]
    # if file exists load it otherwise create a file using convert_to_Tensorfile_input_file.convert    
    if os.path.exists('data/performance_NN.pkl'):
        print('\nperformance_NN.pkl file found.')
        foilsframe = pd.read_pickle('data/performance_NN.pkl')
    else:
        print('NN ready lift file not found trying to create..')
        foilsframe          = convert_to_Tensorflow_input_file.convert(baseframe, label='lift')
        foilsframe['99']    = foilsframe['lift']
        del foilsframe['lift']
        #convert features into np.float32 and labels also to np.float32
        COLUMNS     = [str(i) for i in list(range(100))]
        for i in COLUMNS:
            foilsframe[i] = foilsframe[i].astype(np.float32)
        foilsframe.to_pickle('data/performance_NN.pkl')
        print('file saved')
    return foilsframe, baseframe

if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.FATAL)
    performance_dict = {}

    prediction_set_length   = 150
    steps                   = 400
    epoch                   = 100
    
    threshold               = 15
    #dataset
    foilsframe, baseframe   = get_pandas_file_for_lift()
    print(len(foilsframe), end=' foils ')
    foilsframe = foilsframe[foilsframe['99']<threshold]
    foilsframe = foilsframe[foilsframe['99']>-threshold]
    print('filtered to',len(foilsframe),' capping at +-',threshold)
    
    # shuffle
    temp                    = shuffle(foilsframe)
    training_set            = temp.head(1000) 
    test_set                = temp.tail(300) 
    prediction_set          = temp.tail(300).sample(prediction_set_length)
    
    # hyperparameters
    hidden_units_list = [[50],[75],[25,25],[75,25]]
    
    for hidden_units in hidden_units_list:
    
    #    hidden_units = [50]
    #    hidden_units = [75]
    #    hidden_units = [25,25]
    #    hidden_units = [75,25]
    
        print('hidden_units',hidden_units)
        
        # regressor
        COLUMNS         = [str(i) for i in list(range(100))]
        FEATURES        = COLUMNS[:-1]
        LABEL           = COLUMNS[-1]
        feature_cols    = [tf.feature_column.numeric_column(k) for k in FEATURES]
        regressor       = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                            hidden_units=hidden_units)
                                                   
        loss_score_         = []
        ww_list             = []
        predictions_list    = []
        for _ in range(0, epoch):
            #train
            regressor.train(input_fn=get_input_fn(training_set), steps=steps)   
            
            #evaluate
            ev = regressor.evaluate(input_fn=get_input_fn(test_set, num_epochs=1, shuffle=False))
            loss_score = ev["loss"]
            loss_score_.append(loss_score)
            print('\nepoch',_,', loss: %6.1f'%loss_score)
            print(hidden_units)
    
            # prediction
            # y for predictions, y_ for the labels.
            y = regressor.predict(input_fn=get_input_fn(prediction_set, num_epochs=1, shuffle=False))
            y = list(y)
            
            y = np.array([i['predictions'][0] for i in y])
            foilindex = np.array(prediction_set.index)
            y_        = prediction_set['99'].values
            
            results             = {}
            results['foils']    = foilindex
            results['y']        = y
            results['y_']       = y_
            
                    
            #network structure
            weights = regressor.get_variable_names()    
            var     = [i for i in weights]
            #    w = tf.contrib.learn.DNNClassifier.get_variable_value(classifier,w_name)
            ww = [regressor.get_variable_value(i) for i in var]
            
            
            ww_list.append(ww)
            predictions_list.append(results)
            
            
            if 0: 
                #debug weights with graphics
                for k, (i,j) in enumerate(zip(var,ww)):
                    if k%2==0:
                        print('%3d %40s %10s'%(k,i,j.shape))
                        if len(j.shape)==2:
                            plt.imshow(j, aspect='auto')
                            plt.show()
            if 0:
                #debug with graphics
                fig,ax = plt.subplots()
                draw_neural_net(ax, 0.02, 0.98, 0.02, 0.98, [99,hidden_units[0],2],colors=ww[1:])
                
    
               
            if 1:
                # error graph
                plt.plot(results['y_'], results['y'],'.', alpha=0.3)
                #plt.plot(y_labels,y_labels, alpha=.3)
                plt.show()
    
        
        
        print('\a')
        plt.plot(loss_score_,'.-', alpha=.4)
        plt.show()
        # update analytics dictinary
        if 1:
            key = str(hidden_units)
            performance_dict[key] = {
                            'epochs'        : epoch,
                            'steps'         : steps,
                            'loss'          : loss_score_,
                            'timestamp'     : str(datetime.datetime.now())[:19],
                            'hidden_units'  : hidden_units,
                            'weights'       : ww_list,
                            'predictions'   : predictions_list,
                            }
        print(performance_dict.keys())
        # save dictionary
        if 1:
            filename = 'data/performance_analytics '+'epochs '+str(epoch)+' steps ' + str(steps)
            save_obj(performance_dict,filename)
            print('performance dictionary file saved @ ',filename)
       
