#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 13:01:01 2022

@author: catherinescott
"""
# v2: changed region names and description to match gen_zscore_GMM_v2
# v3: improving naming to make it clearer and more automated. Added test run parameters

#version control:
# v1: copied from v3 of the R1BP version (doesnt remove supercontrols)

import numpy as np
import pandas as pd
import os
import pySuStaIn
import matplotlib.pyplot as plt
import pickle
from pathlib import Path


#determining regions and biomarkers to include in modelling
include_regions = ['frontal','parietal','precuneus','occipital','temporal','insula'] #['frontal','parietal','precuneus','occipital','temporal','insula']#['composite','frontal','parietal','occipital','temporal','insula']#['frontal','parietal','temporal','insula','occipital']
include_biomarkers = ['flow'] #['flow', 'amyloid']
cmmt = ''

# test or run
test_run = 'run' # either 'test' or 'run' determines SuStaIn settings

#remove subjects which were previously put in stage 0?
remove_zero_subs = 'yes'#'yes' #either yes or no

## data in--------------------------------------------------------------------
#descriptions to use for input and output data

in_desc = '1946AVID2YOADSUVR_v1'#'1946-srtm-cleanandAVID27'
if remove_zero_subs=='yes':
    out_desc = in_desc+'-GMM_'+'_'.join(include_biomarkers)+'_'+test_run+'_'+cmmt+'removezero_v1'
else:
    out_desc = in_desc+'-GMM_'+'_'.join(include_biomarkers)+'_'+test_run+'_'+cmmt+'_v1'


region_names=[]
#create list to take values from csv files
for b in include_biomarkers:
    for rg in include_regions:
    
        region_names.append(rg+'_'+b+'_z')
        
#reading in the z-score data
datapath = '/Users/catherinescott/Documents/SuStaIn_out'
csvfile_in = datapath+'/genZscore_out/zscore_allregions_2component_'+in_desc+'-amyloid_co97.5th.csv'


#load in data (size M subjects by N biomarkers, data must be z-scored)
#doesnt skip the header row so that the biomarkers can be ordered according to region_names
#(header row removed in conversion to numpy)

# if we want to get rid of subjects in stage zero
if remove_zero_subs=='yes':
    df_subjs_to_drop = pd.read_csv('/Users/catherinescott/Documents/SuStaIn_out/SuStaIn_crossvalandanalysis/1946AVID2YOADSUVR_v1-GMM_flow_amyloid_run__v1stagezerosubjs.csv')
    df_all = pd.read_csv(csvfile_in)
    cond = df_all['Subject'].isin(df_subjs_to_drop['Subject'])
    df_all.drop(df_all[cond].index, inplace = True)
    #update name of csv file to read in
    csvfile_in = csvfile_in[:len(csvfile_in)-4] + '_removezero' + csvfile_in[len(csvfile_in)-4:]
    #save new csv
    df_all.to_csv(csvfile_in)
    
df = pd.read_csv(csvfile_in, usecols=region_names)[region_names]
data = df.to_numpy()
#remove nans
data = data[~np.isnan(data).any(axis=1)]

#output naming and folders
dataset_name = out_desc
output_folder = os.path.join(datapath,'run_SuStaIn_GMM',dataset_name)
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)


##set params------------------------------------------------------------------
z_lims_csv = datapath+'/genZscore_out/zmax_allregions_2component_'+in_desc+'-amyloid_co97.5th.csv' 
z_lims_df = pd.read_csv(z_lims_csv)
#only include the listed regions
z_lims_df = z_lims_df[z_lims_df['Region'].isin(include_regions)]
#reset the index
z_lims_df.reset_index(inplace=True)

# Z_vals: The set of z-scores to include for each biomarker
# size N biomarkers by Z z-scores
Z_vals = np.zeros([len(region_names),3])
idx = 0
for b in include_biomarkers:
    col_name = b+' z'
    z_vals = z_lims_df.loc[:,col_name]
    #loop through values in the column
    for i in range(len(z_vals)):
        z_val_i = str(z_vals[i]).replace('[','').replace(']','')
        z_val_i = np.fromstring(z_val_i,sep=',')
        for j in range(len(z_val_i)):
            Z_vals[idx,j] = z_val_i[j]
        idx = idx+1

# Z_vals = np.array([[1,2]]*np.shape(data)[1]) #for testing

# Z_max: the maximum z-score reached at the end of the progression
# size N biomarkers by 1
#Z_max = np.percentile(data,95,axis=0).transpose() #for testing
Z_max = []#np.zeros(np.shape(data)[1])
for b in include_biomarkers:
    #Z_max = np.append(Z_max,z_lims_df.loc[:,b+' z_max'].to_numpy)
    Z_max.append(z_lims_df.loc[:,b+' z_max'].tolist())

Z_max = np.array(Z_max).flatten()


# Input the settings for z-score SuStaIn
N_S_max = 3  #max number of subtypes to fit
if test_run == 'test':
    
    print('Running with test params:') 
    N_startpoints = 10 #25 recommended, 10 for testing
    N_iterations_MCMC = int(1e4) #int(1e5) or int(1e6) recommended, 1e4 for testing
    print(str(N_startpoints)+' starting points, '+str(N_iterations_MCMC)+' MCMC iterations')
elif test_run == 'run': 
    print('Running with run params:') 
    N_startpoints = 25 #25 recommended, 10 for testing
    N_iterations_MCMC = int(1.5e5) #int(1e5) or int(1e6) recommended, 1e4 for testing
    print(str(N_startpoints)+' starting points, '+str(N_iterations_MCMC)+' MCMC iterations')
else:
    print('Test or run not given, assume test...')
    N_startpoints = 10 #25 recommended, 10 for testing
    N_iterations_MCMC = int(1e4) #int(1e5) or int(1e6) recommended, 1e4 for testing
    print(str(N_startpoints)+' starting points, '+str(N_iterations_MCMC)+' MCMC iterations')
    
SuStaInLabels = region_names

# Run SuStaIn ---------------------------------------------------------------
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

# plotting results -----------------------------------------------------------

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
    #_ = plt.suptitle('SuStaIn output')
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


    