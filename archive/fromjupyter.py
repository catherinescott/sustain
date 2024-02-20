#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 17:04:58 2022

@author: catherinescott
"""
# import the python packages needed to generate simulated data for the tutorial
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import sklearn.model_selection
import pandas as pd
import pylab
import sys
import pySuStaIn

# this needs to point to wherever the sim folder inside pySuStaIn is on your computer
sys.path.insert(0,'/Users/catherinescott/git_code/pySuStaIn/sim/')
# if you're running the notebook from within the existing structure you can use
# sys.path.insert(0,'../sim/')
from simfuncs import generate_random_Zscore_sustain_model, generate_data_Zscore_sustain

N                       = 5         # number of biomarkers

SuStaInLabels           = []
for i in range(N):
        SuStaInLabels.append( 'Biomarker '+str(i)) # labels of biomarkers for plotting
        
Z_vals                  = np.array([[1,2,3]]*N)     # Z-scores for each biomarker
Z_max                   = np.array([5]*N)           # maximum z-score

# To demonstrate how to set different biomarkers to have different z-scores,
# set biomarker 0 to have z-scores of 1 and 2 only and a maximum of 3
# to do this change the corresponding row of Z_vals to read 1 2 0
# and change the corresponding row of Z_max to 3
Z_vals[np.array(0),np.array(2)] = 0
Z_max[np.array(0)] = 3

# and set biomarker 2 to have a z-score of 1 only and a maximum of 2
# to do this change the corresponding row of Z_vals to read 1 0 0 
# and change the corresponding row of Z_max to 2 
Z_vals[np.array(2),np.array([1,2])] = 0
Z_max[np.array(2)] = 2


# generate a random sequence for the linear z-score model
gt_sequence             = generate_random_Zscore_sustain_model(Z_vals,
                                                        1)

# ignore this part, it's only necessary so that the generate_data_sustain function
# can be used in this demo setting
gt_stages = np.array([0])
gt_subtypes = np.array([0])

# this code generates data from z-score sustain 
# - here i've just output the z-score model itself rather than any datapoints
_, _, gt_stage_value = generate_data_Zscore_sustain(gt_subtypes,
                                             gt_stages,
                                             gt_sequence,
                                             Z_vals,
                                             Z_max)

# ignore this part, just calculates some parameters of sustain to output below
stage_zscore            = np.array([y for x in Z_vals.T for y in x])
stage_zscore            = stage_zscore.reshape(1,len(stage_zscore))
IX_select               = stage_zscore>0
stage_zscore            = stage_zscore[IX_select]
stage_zscore            = stage_zscore.reshape(1,len(stage_zscore))
num_zscores             = Z_vals.shape[1]
IX_vals                 = np.array([[x for x in range(N)]] * num_zscores).T
stage_biomarker_index   = np.array([y for x in IX_vals.T for y in x])
stage_biomarker_index   = stage_biomarker_index.reshape(1,len(stage_biomarker_index))
stage_biomarker_index   = stage_biomarker_index[IX_select]
stage_biomarker_index   = stage_biomarker_index.reshape(1,len(stage_biomarker_index))

# print out some of the values and plot a picture of the model
print('Simulated sequence:',(gt_sequence.astype(int).flatten()))
print('At the beginning of the progression (stage 0) the biomarkers have scores of 0')
print('At the stages:',1+np.arange(np.array(stage_zscore).shape[1]))
print('the biomarkers:',stage_biomarker_index[:,gt_sequence.astype(int).flatten()].flatten())
print('reach z-scores of:',stage_zscore[:,gt_sequence.astype(int).flatten()].flatten())
print('At the end of the progression (stage',np.array(stage_zscore).shape[1]+2,') the biomarkers reach scores of:',Z_max)
print('The z-score model assumes individuals belong to some unknown stage of this progression,')
print('with gaussian noise with a standard deviation of 1 for each biomarker')

temp_stages = np.array(range(np.array(stage_zscore).shape[1]+2))
for b in range(N):
    ax = plt.plot(temp_stages, gt_stage_value[b,:,:])

_ = plt.xlabel('SuStaIn stage')    
_ = plt.ylabel('Z-score')    
_ = plt.legend(SuStaInLabels)
_ = plt.title('Figure 1')