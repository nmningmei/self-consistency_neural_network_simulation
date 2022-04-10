#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 11:46:44 2022

@author: nmei
"""
import os,gc
import numpy as np

from common_functions import garbor_generator

from joblib import Parallel,delayed

if __name__ == "__main__":
    # number of pixels per axis
    image_size = 128
    # 10 lambda values for generating the gabors
    lamdas = np.linspace(4,32,32-4+1)
    # 20 degrees between -89 to 89 degree
    thetaRads = np.linspace(-89,89,100)
    # plot of the generated gabors
    figure_dir = '../data/gabors'
    if not os.path.exists(figure_dir):
        os.mkdir(figure_dir)
    
    gc.collect()
    for lamda_ in lamdas:
        # let's parallelize the for-loop
        Parallel(n_jobs = -1,verbose = 1)(delayed(garbor_generator)(**{
                        'image_size':image_size,
                        'lamda':lamda_,
                        'thetaRad_base':thetaRad_base,
                        'figure_dir':figure_dir}) for thetaRad_base in thetaRads)
    gc.collect()