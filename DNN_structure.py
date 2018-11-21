# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 10:48:44 2017

@author: myaman
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.utils import shuffle

import os, sys, pickle, datetime
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import foil, convert_to_Tensorflow_input_file
import matplotlib.cm as cm


def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def get_input_fn(data_set, num_epochs=None, shuffle=True):
      return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
      y=pd.Series(data_set[LABEL].values),
      num_epochs=num_epochs,
      shuffle=shuffle)
       
def get_pandas_file_for_structure():
    #if file exists load it otherwise create a file using convert_to_Tensorfile_input_file.convert
    FOIL_DATABASE_FILE = 'data/rawbase_with_structure_and_lift.pkl'
    baseframe           = foil.get_foils_frame(FOIL_DATABASE_FILE)
    #baseframe           = baseframe[baseframe['use']==True]
    if os.path.exists('data/structure_NN.pkl'):
        print('\nstructure_NN.pkl file found\n')
        foilsframe = pd.read_pickle('data/structure_NN.pkl')
    else:
        print('NN ready file not found trying to create..')
        foilsframe          = convert_to_Tensorflow_input_file.convert(baseframe, label='is_directed')
        foilsframe['99']    = foilsframe['is_directed']
        del foilsframe['is_directed']
        #convert features into np.float32 and labesl to np.int32
        
        COLUMNS     = [str(i) for i in list(range(100))]

        for i in COLUMNS[:-1]:
            foilsframe[i] = foilsframe[i].astype(np.float32)
        i = str(99)
        foilsframe[i] = foilsframe[i].astype(np.int32)
        foilsframe.to_pickle('data/structure_NN.pkl')
        print('file saved')
    return foilsframe, baseframe

def draw_foil(baseframe,n):
    plt.rcParams['figure.figsize'] = [10.0, 10.0]
    plt.axes().set_aspect('equal')
    plt.plot([-.10,1.10],[-0.15,0.15],'.', alpha=0.1)
    plt.plot(baseframe['x_nn'][n],baseframe['y_nn'][n],'.-')
    return None


if __name__ == '__main__':
    
    tf.logging.set_verbosity('FATAL')
    
    
    foilsframe, baseframe   = get_pandas_file_for_structure()
    # shuffle
    temp                    = shuffle(foilsframe)
    training_set            = temp.head(1094) 
    test_set                = temp.tail(300) 
    
    #d = load_obj('data/structure_analytics')
    
    COLUMNS     = [str(i) for i in list(range(100))]
    FEATURES    = COLUMNS[:-1]
    LABEL       = COLUMNS[-1]
    
    hidden_units = [50,25]
    #hidden_units = [15,15]
    #hidden_units = [75]
    #hidden_units = [25]
    
    ## DNN classifier
    training_steps = 200
    feature_columns = [tf.feature_column.numeric_column(k) for k in FEATURES]
    classifier      = tf.estimator.DNNClassifier(
                                            feature_columns = feature_columns,
                                            hidden_units    = hidden_units,
                                            n_classes       = 2)
    loss_list       = []
    accuracy_list   = []
    divdifs_list    = []
    ww_list         = []
        
    
    for _ in range(0,12):
        
        
        ## train
        classifier.train(input_fn=get_input_fn(training_set), steps=training_steps)
        
        ## evaluate
        accuracy_score = classifier.evaluate(input_fn=get_input_fn(test_set,num_epochs=1, shuffle=False))
        
        
        # analytics
        print('\n')
        print('%25s   %5d'%('iteration',_))
        for __ in accuracy_score:
            print('%25s   %7.2f'%(__, accuracy_score[__]))
 
        ## predictions
        p  = classifier.predict(input_fn=get_input_fn(test_set,num_epochs=1, shuffle=False))
        k  = list(p)
        kk = pd.DataFrame(k)
        kk.index=test_set.index
        kk['div'] = baseframe['div']
        predictions = np.array([i[0] for i in kk['class_ids'].values])
        true_values = test_set['99'].values
        kk['predictions']=predictions
        kk['true_values']=true_values
        kk['win'] = (kk['true_values']==kk['predictions'])
        wins = kk['win']
        #print(sum(kk['win']))
        #print(predictions==true_values)
        print('%25s   %5d'%('wins',sum(predictions==true_values)))
        dif_ = []
        for i_,row in kk.iterrows():
            p1 = row['probabilities'][0]
            p2 = row['probabilities'][1]
            dif_.append(np.abs(p1-p2))
        kk['dif'] = dif_
        kk= kk.sort_values('dif')
        difs = kk['dif'].values
        div = kk['div'].values
        
        
        #network structure
        weights = classifier.get_variable_names()    
        var     = [i for i in weights]
        #    w = tf.contrib.learn.DNNClassifier.get_variable_value(classifier,w_name)
        ww = [classifier.get_variable_value(i) for i in var]
        ww_list.append(ww)
        
    
        if 0:
            fig,ax = plt.subplots()
            draw_neural_net(ax, 0.02, 0.98, 0.02, 0.98, [99,hidden_units[0],2],colors=ww[1:])

        accuracy_list.append(accuracy_score['accuracy'])
        loss_list.append(accuracy_score['loss'])
        divdifs_list.append([difs, div, wins,np.array(kk.index)])
        
        # accuracy plot
        if 1:
            fig = plt.figure()
            plt.rcParams['figure.figsize'] = [8.0, 4.0]
            ax1 = fig.add_axes([0,0,0.5,1])
            ax2 = fig.add_axes([0.6,0,0.55,1])
            ax3 = fig.add_axes([0.8,0.8,0.1,0.1], frame_on=False)
            ax3.set(xticks=[],yticks=[])    
            ax3.set(ylim=(0.60,1.05))
            ax33 = ax1.twinx()
            ax33.set_ylabel('sin', color='r')
            ax33.tick_params('y', colors='r')
            #ax33.set(ylim=(0,30))
            if len(accuracy_list)>1:
                ax1.plot(accuracy_list,'.-')
                ax33.plot(loss_list,'.-')
                
                #
                plt.show()
        
        # prediction plot
        if 0:
            ax2.scatter(kk['dif'],kk['div'], c=kk['win'], alpha=0.7)
            ax2.legend()
            ax2.plot([0,1],[5.6,5.6], lw=30, alpha=.2)
            ax2.set_title("Assurance")
            ax2.set_xlabel("Probability")
            ax2.set_ylabel("pointedness")
            
       
            #inset foil
            ax3.plot([-.10,1.10],[-0.15,0.15],'.', alpha=0.1)
            x__, y__    = baseframe['x_nn'][100],baseframe['y_nn'][100]
            n_          = 5
            s_          = 28
            ax3.plot(x__,y__)
            plt.show()
            

        if 0:
    
            for i,j in zip(var, ww):
                print()
                print('%25s %s'%(i,str(j.shape)))
                print()
                if len(j.shape)==1:
                    j=j.reshape(j.shape[0],-1)
                plt.imshow(j.transpose(), cmap='jet', vmin=j.min(),vmax=j.max(), interpolation='nearest')
                #plt.colorbar()
                plt.show()
            
        
        print('\a')
        
    print('\a')
    print('\a')
    
    keystring       = str(hidden_units) + '_'+str(datetime.datetime.now())[:19]
    d[keystring]    = {'hidden_units':hidden_units,
                     'training':accuracy_list,
                     'loss':loss_list,
                     'test':divdifs_list,
                     'weights':ww_list,
                     'timestamp':str(datetime.datetime.now())[:19],
                     'training_step':training_steps}
    
    if 0:
        save_obj(d,'structure_analytics')
        print('dictionary file saved')
    
    
    if 0:
        for _,(i,j,k) in enumerate(divdifs_list):
            plt.plot([0,1],[5.6,5.6], lw=20, alpha=.02)
            if _% 4 == 0:
                plt.scatter(i,j,c=k, alpha=1.0+0*_/len(divdifs_list)/10, s=10)
                plt.show()

    '''
    #predictions
    #new_samples = np.array([[6.4, 3.2, 4.5, 1.5],[5.8, 3.1, 0.0, 1.7]], dtype=np.float32)
    #predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":new_samples},
    #                                                      num_epochs=1,
    #                                                      shuffle=False)
    #predictions = list(classifier.predict(input_fn= predict_input_fn))
    #for p in predictions:
    #   print(p['classes'])
    '''