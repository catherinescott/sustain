#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:42:32 2024

@author: catherinescott
"""

#this script compares the z-scores generated using the GMM method and SC method.

import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats
import numpy as np

#need to loop over ref region, partial volume correction, timepoint
ref_regions = ['gm-cereb','cereb'] #['cereb']#
PVC_flags = ['pvc-' ,'']#['pvc-' ,'']
data_merge_opts = ['baseline', 'baselineplus', 'followupplus', 'all']  #['followupplus', 'baseline', 'baselineplus', 'all'] 
#include_biomarker_list = [['amyloid','flow'],['flow'],['amyloid']]#[['flow'],['amyloid'],['flow','amyloid']]
params = ['amyloid','flow']


region_names =['composite','frontal','parietal','precuneus','occipital','temporal','insula']

# for figures
row_cnt = 2
plot_across = round(len(region_names)/row_cnt)

zscore_basepath = '/Users/catherinescott/Documents/python_IO_files/sustain_test/SuStaIn_out/genZscoremodsel_out/'

# then for a given method it will plot all regions individually, with SC on the x axis and GMM on the y axis
# and will calculate the paired t-test and put that in the title to see whether it is the same or not.
# could potentally save all of the z-scores and run the paired t-test on the whole lot

for ref_region in ref_regions:
    for PVC_flag in PVC_flags:
        for data_merge_opt in data_merge_opts:
            for param in params:
                #load in the data
                SC_zscore_csv_path = zscore_basepath+PVC_flag+ref_region+'/zscore_allregions_1946AVID2YOADSUVR_v1-'+PVC_flag+ref_region+'_'+data_merge_opt+'_SC.csv'
                GMM_zscore_csv_path = zscore_basepath+PVC_flag+ref_region+'/zscore_allregions_1946AVID2YOADSUVR_v1-'+PVC_flag+ref_region+'_'+data_merge_opt+'.csv'
                SC_zscore_DF = pd.read_csv(SC_zscore_csv_path)
                GMM_zscore_DF = pd.read_csv(GMM_zscore_csv_path)
                
                figures_folder = zscore_basepath+'/figures/GMM_SC_z_compare/'+PVC_flag+ref_region
                if not os.path.exists(figures_folder):
                    os.makedirs(figures_folder)
                
                #create the figure                                   
                fig, ax = plt.subplots(2,plot_across)
                fig.set_figwidth(20)
                fig.set_figheight(10)
                fig_count = 0
                
                # rename columns so the methods can be distinguished
                SC_zscore_DF = SC_zscore_DF.add_suffix('_SC')
                SC_zscore_DF.rename(columns={"Subject_SC": "Subject"}, inplace=True)
                
                GMM_zscore_DF = GMM_zscore_DF.add_suffix('_GMM')
                GMM_zscore_DF.rename(columns={"Subject_GMM": "Subject"}, inplace=True)
                
                df_merged = pd.merge(GMM_zscore_DF, SC_zscore_DF, on=['Subject'])
                
                for region in region_names:
                    #print(region)
                    if fig_count >=plot_across:
                        row_no=1
                        col_no=fig_count-plot_across
                    else:
                        row_no=0
                        col_no = fig_count
                    

                    
                    variable_nm = region+'_'+param+'_z'
                    
                    if variable_nm+'_GMM' in df_merged.columns and variable_nm+'_SC' in df_merged.columns:
                        #print('region INCLUDED: '+region)
                        GMM_data=df_merged[[variable_nm+'_GMM']].values
                        SC_data=df_merged[[variable_nm+'_SC']].values
                       
                        min_plot = min(min(GMM_data),min(SC_data))
                        max_plot = max(max(GMM_data),max(SC_data))
                        
                        #[statistic, p_value]=stats.ttest_rel(SC_data, GMM_data)
                        #[statistic, p_value]=stats.wilcoxon(df_merged[variable_nm+'_GMM'], df_merged[variable_nm+'_SC'])
                        [statistic, p_value]=stats.wilcoxon(np.round(df_merged[variable_nm+'_GMM']-df_merged[variable_nm+'_SC'],1))
                        
                        ax[row_no,col_no].plot([min_plot,max_plot], [min_plot,max_plot], color='k', linestyle='dashed',linewidth=2)
                        ax[row_no,col_no].scatter(SC_data, GMM_data, alpha=0.2, label = 'p= '+str(np.round(p_value,10)))
                        
                        ax[row_no,col_no].legend()

                        #no_pvc_data = np.squeeze(no_pvc_data)
                        #pvc_data = np.squeeze(pvc_data)
                        #ax[row_no,col_no].plot([min_plot[0],max_plot[0]], np.poly1d(np.polyfit(no_pvc_data, pvc_data, 1))([min_plot[0],max_plot[0]]),label=variable_type.replace("_", "")+'fit')
                    #else:
                        #print('region EXCLUDED: '+region)
                    #ax[row_no,col_no].legend()
                    #print('region: '+region)
                    #print('fig:'+str(fig_count))
                    #print('Row:'+str(row_no)+', Col:'+str(col_no))
                    ax[row_no,col_no].set_title('param:,'+param+'ref: '+ref_region+', region: '+region)
                    ax[row_no,col_no].set(xlabel='Super Control (SC) z-score', ylabel='Gaussian mixture model (GMM) z-score')
    
                    fig_count = fig_count+1
                
            
                fig.savefig(os.path.join(figures_folder,ref_region+'_'+PVC_flag+data_merge_opt+'_'+param+'.pdf'))                 
                    