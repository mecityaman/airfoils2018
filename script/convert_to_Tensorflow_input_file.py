# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 11:54:52 2017

@author: myaman
"""


import foil
import time

import tensorflow as tf
import numpy as np
import pandas as pd


def convert(foilsframe, label='lift'):

    # takes in a pandas foil that has a y_nn columns that are
    # exactly 99 elements long and 
    # creates a new frame that has each of this elements as a column
    # and adds a last column ('label') to shows target information.

    n_of_features_cols  = 99    
    columns             = [str(i) for i in list(range(n_of_features_cols))]
    aframe              = pd.DataFrame(columns=columns)

    for i,j in foilsframe.iterrows():
        temp = j['y_nn']
        for _ in range(n_of_features_cols):
            aframe.set_value(i,str(_),temp[_])
       
        aframe.set_value(i,label,j[label])
        print(i, j[label])
    
    return aframe




if __name__ == '__main__':


    FOIL_DATABASE_FILE = 'foil data/base.pkl'
    foilsframe         = foil.get_foils_frame(FOIL_DATABASE_FILE)
    foilsframe         = foilsframe[foilsframe['use']==True]
    
    newframe = convert (foilsframe, label='is_directed')        
    print(len(newframe))
    print(newframe.head())
    
    #newframe.to_pickle('liftforNN.pkl')
