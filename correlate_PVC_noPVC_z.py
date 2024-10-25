#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 16:12:09 2024

@author: catherinescott
"""
#trying to diagnose why PVc data doesnt wrok but non pvc does

import os
import pandas as pd
import matplotlib.pyplot as plt

region_names =['composite','frontal','parietal','precuneus','occipital','temporal','insula']
variable_types = ['z','suvr']

ref_regions = ['cereb','gm-cereb']
row_cnt = 2
plot_across = round(len(region_names)/row_cnt)

data_folder = '/Users/catherinescott/Documents/python_IO_files/sustain_test/SuStaIn_out/genZscoremodsel_out/'
figures_folder = data_folder+'figures/correlate_PVC_no_PVC/'

if not os.path.exists(figures_folder):
    os.makedirs(figures_folder)


for chosen_variable in variable_types:
    
    variable_type = '_amyloid_'+chosen_variable #'_amyloid_z'
    variable_type_2 ='_flow_'+chosen_variable#'_flow_z'
    
    for ref_region in ref_regions:
        
            
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
                
            csv_nopvc = '/Users/catherinescott/Documents/python_IO_files/sustain_test/SuStaIn_out/genZscoremodsel_out/'+ref_region+'/zscore_allregions_1946AVID2YOADSUVR_v1-'+ref_region+'_all_SC.csv'
        
            df_no = pd.read_csv(csv_nopvc)
            df_no = df_no.add_suffix('_nopvc')
            df_no.rename(columns={"Subject_nopvc": "Subject"}, inplace=True)
            
            csv_pvc = '/Users/catherinescott/Documents/python_IO_files/sustain_test/SuStaIn_out/genZscoremodsel_out/pvc-'+ref_region+'/zscore_allregions_1946AVID2YOADSUVR_v1-pvc-'+ref_region+'_all_SC.csv'
            df_pvc = pd.read_csv(csv_pvc)
            df_pvc = df_pvc.add_suffix('_pvc')
            df_pvc.rename(columns={"Subject_pvc": "Subject"}, inplace=True)
            
            df_merged = pd.merge(df_no, df_pvc, on=['Subject'])
            
            variable_nm = region+variable_type#'occipital_amyloid_z'
            
            
            if variable_nm+'_nopvc' and variable_nm+'_pvc' in df_merged.columns:
                
                no_pvc_data=df_merged[[variable_nm+'_nopvc']].values
                pvc_data=df_merged[[variable_nm+'_pvc']].values
               
                min_plot = min(min(no_pvc_data),min(pvc_data))
                max_plot = max(max(no_pvc_data),max(pvc_data))
                ax[row_no,col_no].plot([min_plot,max_plot], [min_plot,max_plot], color='k', linestyle='dashed',linewidth=2)
                ax[row_no,col_no].scatter(no_pvc_data, pvc_data, label=variable_type.replace("_", ""))
            
            
            
            variable_nm_2= region+variable_type_2#'occipital_amyloid_z'
            
            if variable_nm_2+'_nopvc' and variable_nm_2+'_pvc' in df_merged.columns:
                            
                no_pvc_data_2=df_merged[[variable_nm_2+'_nopvc']].values
                pvc_data_2=df_merged[[variable_nm_2+'_pvc']].values
                ax[row_no,col_no].scatter(no_pvc_data_2, pvc_data_2, label=variable_type_2.replace("_", ""))
                
                
            ax[row_no,col_no].legend()
            #ax[row_no,col_no].set_title(variable_nm+' ref: '+ref_region)
            ax[row_no,col_no].set_title('ref: '+ref_region)
            ax[row_no,col_no].set(xlabel='no PVC', ylabel='PVC')
            #ax.xlabel("no PVC")
            #ax.ylabel("PVC")
            #ax.show()
            fig_count = fig_count+1
        
    
        fig.savefig(os.path.join(figures_folder,ref_region+variable_type+variable_type_2+'.pdf'))
