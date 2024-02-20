#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 13:01:01 2022

@author: catherinescott
"""
import numpy as np
import pandas as pd
import os
#import nibabel as nib
import pySuStaIn
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
#from get_biomarker_order import get_biomarker_order

## data in--------------------------------------------------------------------
#descriptions to use for input and output data
in_desc = '1946-srtm-clean'
out_desc = in_desc+'-longswitch'

#reading in the data
csvfile_in = 'subj-zscores-'+in_desc+'.csv'
#using a subset of the regions
#region_names =['insula_R1_z','temporal_R1_z','frontal_R1_z','parietal_R1_z','occipital_R1_z']
region_names =['frontal_R1_z','parietal_R1_z','insula_R1_z','temporal_R1_z','occipital_R1_z']
#load in data (size M subjects by N biomarkers, data must be z-scored)
#doesnt skip the header row so that the biomarkers can be ordered according to region_names
#(header row removed in conversion to numpy)
df = pd.read_csv(os.path.join(os.getcwd(),csvfile_in), usecols=region_names)[region_names]
data = df.to_numpy()

##set params------------------------------------------------------------------
# Z_vals: The set of z-scores to include for each biomarker
# size N biomarkers by Z z-scores
Z_vals = np.array([[1,2]]*np.shape(data)[1])
# Z_max: the maximum z-score reached at the end of the progression
# size N biomarkers by 1
Z_max = np.percentile(data,95,axis=0).transpose()

# Input the settings for z-score SuStaIn

N_startpoints = 25 #25 recommended, 10 for testing
N_S_max = 2  #max number of subtypes to fit
N_iterations_MCMC = int(1e6) #int(1e5) or int(1e6) recommended, 1e4 for testing
dataset_name = 'R1'+out_desc
output_folder = os.path.join(os.getcwd(), dataset_name)
SuStaInLabels = region_names

sustain_input = pySuStaIn.ZscoreSustain(data,
                              Z_vals,
                              Z_max,
                              SuStaInLabels,
                              N_startpoints,
                              N_S_max, 
                              N_iterations_MCMC, 
                              output_folder, 
                              dataset_name, 
                              False)

if not os.path.isdir(output_folder):
    os.mkdir(output_folder)
    
    # runs the sustain algorithm with the inputs set in sustain_input above
samples_sequence,   \
samples_f,          \
ml_subtype,         \
prob_ml_subtype,    \
ml_stage,           \
prob_ml_stage,      \
prob_subtype_stage  = sustain_input.run_sustain_algorithm()

# #plotting the positional variance diagram
# _ = plt.figure(3)
# pySuStaIn.ZscoreSustain._plot_sustain_model(sustain_input,samples_sequence,samples_f,len(data),biomarker_labels=SuStaInLabels)
# _ = plt.suptitle('SuStaIn output')
# plt.savefig(os.path.join(output_folder,'SuStaIn_output'+desc+'.pdf'))

# go through each subtypes model and plot MCMC samples of the likelihood
for s in range(N_S_max):
    pickle_filename_s           = output_folder + '/pickle_files/' + dataset_name + '_subtype' + str(s) + '.pickle'
    pickle_filepath             = Path(pickle_filename_s)
    pickle_file                 = open(pickle_filename_s, 'rb')
    loaded_variables            = pickle.load(pickle_file)
    samples_likelihood          = loaded_variables["samples_likelihood"]
    samples_sequence            = loaded_variables["samples_sequence"]
    samples_f                   = loaded_variables["samples_f"]
    pickle_file.close()

    #MCMC plots
    _ = plt.figure(0)
    _ = plt.plot(range(N_iterations_MCMC), samples_likelihood, label="subtype" + str(s))
    _ = plt.legend(loc='upper right')
    _ = plt.xlabel('MCMC samples')
    _ = plt.ylabel('Log likelihood')
    _ = plt.title('MCMC trace')
    plt.savefig(os.path.join(output_folder,'MCMC_subtype_'+ str(s)+out_desc+'.pdf'))
    
    #histogram plots
    _ = plt.figure(1)
    _ = plt.hist(samples_likelihood, 80, label="subtype" + str(s))
    _ = plt.legend(loc='upper right')
    _ = plt.xlabel('Log likelihood')  
    _ = plt.ylabel('Number of samples')  
    _ = plt.title('Figure 6: Histograms of model likelihood')
    plt.savefig(os.path.join(output_folder,'hist_subtype_'+ str(s)+out_desc+'.pdf'))
    
    #plotting the positional variance diagram
    _ = plt.figure(2)
    pySuStaIn.ZscoreSustain._plot_sustain_model(sustain_input,samples_sequence,samples_f,len(data),biomarker_labels=SuStaInLabels)
    _ = plt.suptitle('SuStaIn output')
    plt.savefig(os.path.join(output_folder,'SuStaIn_output_subtype_'+ str(s)+out_desc+'.pdf'))
    

# #plotting the positional variance diagram
# _ = plt.figure(3)

# something = get_biomarker_order(samples_sequence,samples_f,len(data),Z_vals,biomarker_labels=SuStaInLabels)
# _ = plt.suptitle('SuStaIn output')
# plt.savefig(os.path.join(output_folder,'SuStaIn_output'+desc+'.pdf'))

# _ = plt.figure(0)
# _ = plt.legend(loc='upper right')
# _ = plt.xlabel('MCMC samples')
# _ = plt.ylabel('Log likelihood')
# _ = plt.title('MCMC trace')
# plt.savefig(os.path.join(output_folder,'MCMC_subtype_'+ str(s)+'_lablled.pdf'))
   
# _ = plt.figure(1)
# _ = plt.legend(loc='upper right')
# _ = plt.xlabel('Log likelihood')  
# _ = plt.ylabel('Number of samples')  
# _ = plt.title('Figure 6: Histograms of model likelihood')
# plt.savefig(os.path.join(output_folder,'hist_subtype_'+ str(s)+desc+'_labelled.pdf'))


    