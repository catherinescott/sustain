#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 15:52:48 2022

@author: catherinescott
"""
import numpy as np
import pandas as pd
import os
import nibabel as nib

#define paths
datapath = '/Users/catherinescott/Documents/1946/pig/JCBFM_data/data'
outpath = '/Users/catherinescott/PycharmProjects/sustain_test'

#get the IDs of all the subjects from the folder names
allsubjects = os.listdir(datapath)
allsubjects= sorted(allsubjects)
#removed unneessary files
allsubjects.pop(0)
allsubjects.pop(-1)

#set subject status: take YOAD as positive ('PT') and 1946 as control ('CL')
subjstatus = []

for i in range(4):
    subjstatus.append('PT')
    
for j in range((len(allsubjects)-4)):
    subjstatus.append('CL')
#write to a dataframe
df = pd.DataFrame({'Subj_ID':allsubjects, 'Status':subjstatus})



#load in image data----------------------------------------------------------

allR1valsarray = []

#loop through all subjects in the list
for k in range(len(allsubjects)):
    imagepath = datapath+'/'+allsubjects[k]
    #load R1 image
    imgR1 = nib.load(imagepath+'/parameter_maps/'+allsubjects[k]+'_sm2_rgnR1_DBv5.nii.gz')
    #print(imagepath)
    #load parcellation
    imgprc = nib.load(imagepath+'/prcl/parc6GMallv3toPET.nii.gz')
    
    #get the unique values in the parcellation
    u, indices =np.unique(imgprc.get_fdata(),return_index=True)
    #remove the 0 at the beginning (which corresponds to the background)
    useindices = indices[1:len(indices)]
    #remove the reference region
    useindices = np.delete(useindices,10)
    
    #find the corresponding values in the R1 image
    R1arr = imgR1.get_fdata()
    R1vals = R1arr.ravel()[useindices]
    
    #add to array
    allR1valsarray.append([R1vals])

#convert to array    
allR1valsarray = np.asarray(allR1valsarray)
allR1valsarray = np.squeeze(allR1valsarray)

region_names =['frontal', 'temporal', 'parietal', 'occipital', 'cingulate', \
               'insula', 'accumbens', 'amygdala', 'brainstem', 'caudate', \
               'cerebellar white', 'cerebral white matter', 'hippocampus', \
               'pallidum', 'putamen', 'thalamus']
    
#add columns to dataframe
for j in range(len(region_names)):
    df[region_names[j]] = allR1valsarray[:,j]
    
#write dataframe to csv file ('/Users/catherinescott/PycharmProjects/sustain_test')
df.to_csv('subj_info.csv')

#required output csv file with a row for each subject,
# columns: binary classification as patient or control, biomarker values


# converting to z score data
# take only control data:
df_ctl = df.drop(df[df.Status == 'PT'].index)

for l in range(2,(len(region_names)+2)):
    
    # compute the mean and standard deviation of the control population
    mean_control = df_ctl[region_names[l-2]].mean()
    std_control = df_ctl[region_names[l-2]].std()
    
    #add the updated z score into the dataframe with '_z' 
    #appended to the column name
    #NOTE: here I multiply it all by -1 as blood flow reduces as the disease progresses
    # need to make sure that I dont do this for amyloid!
    df[region_names[l-2] + '_z'] = -1*(df[region_names[l-2]]-mean_control)/std_control

#write all results to file
df.to_csv('subj_zscores.csv')