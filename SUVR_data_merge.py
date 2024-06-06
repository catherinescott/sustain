#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 10:46:30 2024

@author: catherinescott
"""

#this codetakes input csv files and merges them
#input: various csv files containing SUVR data
#function: combines all the available csvs to create a) a normal database and b) a database for sustain modelling
#outputs: normal database csv, sustain modelling csv


import os

# parameters to set/test:----------------------------------------------------
param = 'amyloid' #'BPnd' #this is the parameter used to define normality
#see data_cols for all params read in
#(assume that we want to generate z-scores for R1 and BPnd (hardcoded))
# composite must be the first region as this is used to define status
region_names =['composite','frontal','parietal','precuneus','occipital','temporal','insula']
components_to_fit = 2
plot_centile = 97.5
cutoff_centile = 97.5


#input/output-----------------------------------------------------------------
#description to add to outputs
version_no = '1'
ref_region = 'gm-cereb' # options: 'cereb', 'gm-cereb'
PVC_flag = '' # options: 'pvc-', ''
desc = '1946AVID2YOADSUVR_v'+version_no+'-'+param+'-'+PVC_flag+ref_region #1946-srtm-cleanandAVID27-'+param
data_merge_opt = 'all' # 'baseline'

#define paths
out_folder = '/Users/catherinescott/Documents/SuStaIn_out'
datapath = '/Users/catherinescott/Documents/python_IO_files/input_csv_files/SUVR_spreadsheets/opt_4i2mm/suvr-'+PVC_flag+'nipet-pct-gif-'+ref_region
outpath = out_folder+'/genZscore_out'
if not os.path.exists(outpath):
    os.makedirs(outpath)