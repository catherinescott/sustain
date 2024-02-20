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
from get_subj_status import get_subj_status
import glob

#description to add to outputs
desc = '1946-srtm-cleanandAVID27'

#define paths
datapath = '/Users/catherinescott/PycharmProjects/sustain_test/csvfiles'
outpath = '/Users/catherinescott/PycharmProjects/sustain_test'

#regions to be included in analysis
region_names = ['frontal','parietal','temporal','insula','occipital']

#load in the csv file containing the srtm parameters for each subject
#assuming that you want to use all the csv files in the datapath folder
datacols=['Subject', 'Session','ROI','R1_srtm','BPnd_srtm','beta']
all_csv_files = glob.glob(os.path.join(datapath, "*.csv"))
df = pd.concat((pd.read_csv(f, skiprows=0,usecols=datacols) for f in all_csv_files), ignore_index=True)

#df = pd.read_csv(os.path.join(datapath,'srtm_stats_20200703_015300.csv'), skiprows=0,usecols=datacols)

# function to determine whether a particular subject should be treated as pt or ctl
# #plot histogram to get a rough cut point
# BPs = df.loc[df['ROI']=='composite' ,'BPnd_srtm']
# #plt.style.use('seaborn')
# fig, ax = plt.subplots(1,1)
# counts, bins, patches= ax.hist(BPs,100)

#add status to original dataframe
df_status = get_subj_status(df, 0.27)
df_complete = df.merge(df_status,how='left',on='Subject')



#for now only use baseline sessions and R1 values for regions in region_names
R1_details = df_complete[df_complete['ROI'].isin(region_names)]
R1_details = R1_details.loc[(R1_details['Session']=='baseline'),['Subject','ROI','R1_srtm','Status']]
#remove nans
R1_details.dropna(inplace=True)

#make regions the column names
R1_table = (R1_details.pivot_table('R1_srtm',['Subject','Status'],'ROI').rename_axis(columns=None)
         .reset_index())

# converting to z score data
# take only control data:
df_ctl = R1_table.drop(R1_table[R1_table.Status == 'PT'].index)

#different format to previous test script, consider whether I want to 
#rearrange the dataframe to match when I test on multiple regions
for l in range(2,(len(region_names)+2)):
    
    # compute the mean and standard deviation of the control population
    mean_control = df_ctl[region_names[l-2]].mean()
    std_control = df_ctl[region_names[l-2]].std()
    
    #add the updated z score into the dataframe with '_z' 
    #appended to the column name
    #NOTE: here I multiply it all by -1 as blood flow reduces as the disease progresses
    # need to make sure that I dont do this for amyloid!
    R1_table[region_names[l-2] + '_R1_z'] = -1*(R1_table[region_names[l-2]]-mean_control)/std_control

# mean_control = df_ctl[datacol].mean()
# std_control = df_ctl[datacol].std()

# #add the updated z score into the dataframe with '_z' 
# #appended to the column name
# #NOTE: here I multiply it all by -1 as blood flow reduces as the disease progresses
# # need to make sure that I dont do this for amyloid!
# R1_details[region_names[0]+'_z']=-1*(R1_details[datacol]-mean_control)/std_control
    
# # for l in range(2,(len(region_names)+2)):
    
# #     # compute the mean and standard deviation of the control population
# #     mean_control = df_ctl[region_names[l-2]].mean()
# #     std_control = df_ctl[region_names[l-2]].std()
    
# #     #add the updated z score into the dataframe with '_z' 
# #     #appended to the column name
# #     #NOTE: here I multiply it all by -1 as blood flow reduces as the disease progresses
# #     # need to make sure that I dont do this for amyloid!
# #     df[region_names[l-2] + '_z'] = -1*(df[region_names[l-2]]-mean_control)/std_control

#write all results to file
R1_table.to_csv('subj-zscores-'+desc+'.csv')