#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 16:07:30 2022

@author: catherinescott
"""
import numpy as np
import pandas as pd
import os
import nibabel as nib
import matplotlib.pyplot as plt

#description to add to outputs
desc = '1946-srtm-clean'

#define paths
datapath = '/Users/catherinescott/Desktop'
outpath = '/Users/catherinescott/PycharmProjects/sustain_test'

#load in the csv file comtaining the srtm parameters for each subject
datacols=['Subject', 'Session','ROI','R1_srtm','BPnd_srtm','beta']
df = pd.read_csv(os.path.join(datapath,'srtm_stats_20200703_015300.csv'), skiprows=0,usecols=datacols)

# #plot histogram to get a rough cut point
# BPs = df.loc[df['ROI']=='composite' ,'BPnd_srtm']
# #plt.style.use('seaborn')
# fig, ax = plt.subplots(1,1)
# counts, bins, patches= ax.hist(BPs,100)
cutoff = 0.35

#label all patients 'PT' if BPnd is greater than cutoff or 'CTL' otherwise
df['Status'] = np.where(df['BPnd_srtm']<=cutoff,'CTL','PT')


#for now only use baseline sessions and composite R1 values

R1_details = df.loc[(df['ROI']==region_names[0]) & (df['Session']=='baseline'),['Subject',datacol,'Status']]
R1_details.dropna(inplace=True)

# converting to z score data
# take only control data:
df_ctl = R1_details.drop(R1_details[R1_details.Status == 'PT'].index)

#different format to previous test script, consider whether I want to 
#rearrange the dataframe to match when I test on multiple regions


mean_control = df_ctl[datacol].mean()
std_control = df_ctl[datacol].std()

#add the updated z score into the dataframe with '_z' 
#appended to the column name
#NOTE: here I multiply it all by -1 as blood flow reduces as the disease progresses
# need to make sure that I dont do this for amyloid!
R1_details[region_names[0]+'_z']=-1*(R1_details[datacol]-mean_control)/std_control
    
# for l in range(2,(len(region_names)+2)):
    
#     # compute the mean and standard deviation of the control population
#     mean_control = df_ctl[region_names[l-2]].mean()
#     std_control = df_ctl[region_names[l-2]].std()
    
#     #add the updated z score into the dataframe with '_z' 
#     #appended to the column name
#     #NOTE: here I multiply it all by -1 as blood flow reduces as the disease progresses
#     # need to make sure that I dont do this for amyloid!
#     df[region_names[l-2] + '_z'] = -1*(df[region_names[l-2]]-mean_control)/std_control

#write all results to file
R1_details.to_csv('subj-zscores-'+datacol+'-'+desc+'.csv')