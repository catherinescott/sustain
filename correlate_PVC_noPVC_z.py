#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 16:12:09 2024

@author: catherinescott
"""
#trying to diagnose why PVc data doesnt wrok but non pvc does

import pandas as pd
import matplotlib.pyplot as plt

region_names =['composite','frontal','parietal','precuneus','occipital','temporal','insula']
variable_type = '_amyloid_z'
ref_region = 'gm-cereb'
row_cnt = 2
plot_across = round(len(region_names)/row_cnt)

fig, ax = plt.subplots(2,plot_across)
fig.set_figwidth(20)
fig.set_figheight(10)
fig_count = 0

for region in region_names:
    
    if fig_count >=plot_across:
        row_no=1
        col_no=fig_count-plot_across
    else:
        row_no=0
        col_no = fig_count
        
    csv_nopvc = '/Users/catherinescott/Documents/python_IO_files/sustain_test/SuStaIn_out/genZscoremodsel_out_a/'+ref_region+'/zscore_allregions_1946AVID2YOADSUVR_v1-'+ref_region+'_baseline.csv'
    df_no = pd.read_csv(csv_nopvc)
    df_no = df_no.add_suffix('_nopvc')
    df_no.rename(columns={"Subject_nopvc": "Subject"}, inplace=True)
    
    csv_pvc = '/Users/catherinescott/Documents/python_IO_files/sustain_test/SuStaIn_out/genZscoremodsel_out_a/pvc-'+ref_region+'/zscore_allregions_1946AVID2YOADSUVR_v1-pvc-'+ref_region+'_baseline.csv'
    df_pvc = pd.read_csv(csv_pvc)
    df_pvc = df_pvc.add_suffix('_pvc')
    df_pvc.rename(columns={"Subject_pvc": "Subject"}, inplace=True)
    
    df_merged = pd.merge(df_no, df_pvc, on=['Subject'])
    
    variable_nm = region+variable_type#'occipital_amyloid_z'
    
    no_pvc_data=df_merged[[variable_nm+'_nopvc']].values
    pvc_data=df_merged[[variable_nm+'_pvc']].values
    
    
    
    ax[row_no,col_no].scatter(no_pvc_data, pvc_data)
    ax[row_no,col_no].set_title(variable_nm+' ref: '+ref_region)
    ax[row_no,col_no].set(xlabel='no PVC', ylabel='PVC')
    #ax.xlabel("no PVC")
    #ax.ylabel("PVC")
    #ax.show()
    fig_count = fig_count+1
