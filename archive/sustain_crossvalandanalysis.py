#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 10:59:14 2022

@author: catherinescott
"""
import numpy as np
import pandas
import os
import seaborn as sns
#import nibabel as nib
import pySuStaIn
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import sklearn.model_selection

in_desc = '1946-srtm-cleanandAVID27'
out_desc = in_desc+'-GMM_R1BP_long' #in_desc+'-GMM_R1BP_long'
dataset_name = 'R1BPnd'+out_desc
output_folder = os.path.join(os.getcwd(), dataset_name)
datapath = '/Users/catherinescott/PycharmProjects/sustain_test/mixturemodel_out'
csvfile_in = datapath+'/zscore_allregions_2component_'+in_desc+'-BPnd_co97.5th.csv'

#determining regions and biomarkers to include in modelling
include_regions = ['frontal','parietal','temporal','insula','occipital']
include_biomarkers = ['R1','BPnd']#['R1', 'BPnd']
region_names=[]
#create list to take values from csv files
for b in include_biomarkers:
    for rg in include_regions:
    
        region_names.append(rg+'_'+b+'_z')

# The SuStaIn output has everything we need. We'll use it to populate our dataframe.

#load in data (size M subjects by N biomarkers, data must be z-scored)
#doesnt skip the header row so that the biomarkers can be ordered according to region_names
#(header row removed in conversion to numpy)
df = pandas.read_csv(os.path.join(os.getcwd(),csvfile_in), usecols=region_names)[region_names]
subjs= pandas.read_csv(os.path.join(os.getcwd(),csvfile_in), usecols=['Subject'])
status= pandas.read_csv(os.path.join(os.getcwd(),csvfile_in), usecols=['Status'])

data = df.to_numpy()
#remove nans
nonNaN_subjects = ~np.isnan(data).any(axis=1)
data = data[nonNaN_subjects]
zdata = pandas.DataFrame(data,columns= list(df.columns))
#subj_IDs = df.loc[:,]
zdata['Subject'] = subjs[nonNaN_subjects].reset_index(drop=True)
zdata['Status'] = status[nonNaN_subjects].reset_index(drop=True)

#replace status ctl vs label as 0 and 1
zdata.replace('CTL',0,inplace=True)
zdata.replace('PT',1,inplace=True)

##set params------------------------------------------------------------------
z_lims_csv = datapath+'/zmax_allregions_2component_'+in_desc+'-BPnd_co97.5th.csv' 
z_lims_df = pandas.read_csv(z_lims_csv)
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

N_startpoints = 25 #25 recommended, 10 for testing
N_S_max = 4  #max number of subtypes to fit
N_iterations_MCMC = int(1e6) #int(1e5) or int(1e6) recommended, 1e4 for testing
dataset_name = 'R1BPnd'+out_desc
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


#---------------------------------------------------------------------

M = len(zdata) 

s=2

pickle_filename_s = output_folder + '/pickle_files/' + dataset_name + '_subtype' + str(s) + '.pickle'
pk = pandas.read_pickle(pickle_filename_s)

# let's take a look at all of the things that exist in SuStaIn's output (pickle) file
pk.keys()

for variable in ['ml_subtype', # the assigned subtype
                 'prob_ml_subtype', # the probability of the assigned subtype
                 'ml_stage', # the assigned stage 
                 'prob_ml_stage',]: # the probability of the assigned stage
    
    # add SuStaIn output to dataframe
    zdata.loc[:,variable] = pk[variable] 

# let's also add the probability for each subject of being each subtype
for i in range(s):
    zdata.loc[:,'prob_S%s'%i] = pk['prob_subtype'][:,i]
zdata.head()

# IMPORTANT!!! The last thing we need to do is to set all "Stage 0" subtypes to their own subtype
# We'll set current subtype (0 and 1) to 1 and 0, and we'll call "Stage 0" individuals subtype 0.

# make current subtypes (0 and 1) 1 and 2 instead
zdata.loc[:,'ml_subtype'] = zdata.ml_subtype.values + 1

# convert "Stage 0" subjects to subtype 0
zdata.loc[zdata.ml_stage==0,'ml_subtype'] = 0


zdata.ml_subtype.value_counts()


biomarkerstring = ' '.join(include_biomarkers)
print('Total subjects included in '+biomarkerstring+': '+str(len(zdata)))

for subtype in range(0,s+2):

    print('for '+biomarkerstring+ ' subtype '+str(subtype)+' n subjects ='+str(len(zdata.loc[zdata['ml_subtype']==subtype,'ml_stage'])))

#As a sanity check, let's make sure all the "controls" were given assigned to low stages by SuStaIn

sns.displot(x='ml_stage',hue='Status',data=zdata,col='ml_subtype')

#And now, let's plot the subtype probabilities over SuStaIn stages to make sure we don't have any crossover events

sns.pointplot(x='ml_stage',y='prob_ml_subtype', # input variables
              hue='ml_subtype',                 # "grouping" variable
            data=zdata[zdata.ml_subtype>0]) # only plot for Subtypes 1 and 2 (not 0)
plt.ylim(0,1) 
plt.axhline(0.5,ls='--',color='k') # plot a line representing change (0.5 in the case of 2 subtypes)

#cross validation

# choose the number of folds - here i've used three for speed but i recommend 10 typically
N_folds = 3

# generate stratified cross-validation training and test set splits
labels = zdata.Status.values
cv = sklearn.model_selection.StratifiedKFold(n_splits=N_folds, shuffle=True)
cv_it = cv.split(zdata, labels)

# SuStaIn currently accepts ragged arrays, which will raise problems in the future.
# We'll have to update this in the future, but this will have to do for now
test_idxs = []
for train, test in cv_it:
    test_idxs.append(test)
test_idxs = np.array(test_idxs,dtype='object')

# perform cross-validation and output the cross-validation information criterion and
# log-likelihood on the test set for each subtypes model and fold combination
CVIC, loglike_matrix     = sustain_input.cross_validate_sustain_model(test_idxs)


# go through each subtypes model and plot the log-likelihood on the test set and the CVIC
print("CVIC for each subtype model: " + str(CVIC))
print("Average test set log-likelihood for each subtype model: " + str(np.mean(loglike_matrix, 0)))

plt.figure(0)    
plt.plot(np.arange(N_S_max,dtype=int),CVIC)
plt.xticks(np.arange(N_S_max,dtype=int))
plt.ylabel('CVIC')  
plt.xlabel('Subtypes model') 
plt.title('CVIC')

plt.figure(1)
df_loglike = pandas.DataFrame(data = loglike_matrix, columns = ["s_" + str(i) for i in range(sustain_input.N_S_max)])
df_loglike.boxplot(grid=False)
plt.ylabel('Log likelihood')  
plt.xlabel('Subtypes model') 
plt.title('Test set log-likelihood across folds')


#Another useful output of the cross-validation that you can look at are positional variance diagrams averaged across cross-validation folds. These give you an idea of the variability in the progression patterns across different training datasets
#this part estimates cross-validated positional variance diagrams
for i in range(N_S_max):
    sustain_input.combine_cross_validated_sequences(i+1, N_folds)
    
    N_S_selected = i+1#2
    
#dont need it to replot the original as I've already saved it
#pySuStaIn.ZscoreSustain._plot_sustain_model(sustain_input,samples_sequence,samples_f,M,subtype_order=(0,1))
#_ = plt.suptitle('SuStaIn output')

    sustain_input.combine_cross_validated_sequences(N_S_selected, N_folds)
    _ = plt.suptitle('Cross-validated SuStaIn output')