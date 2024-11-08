#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 12:39:39 2024

@author: catherinescott
"""
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

region_names =['composite','frontal','parietal','precuneus','occipital','temporal','insula']
#components_to_fit = 2
plot_centile = 97.5
cutoff_centile = 97.5

# ref_regions = ['cereb']#['cereb', 'gm-cereb']
# PVC_flags = ['pvc-' ,'']
# data_merge_opts = ['baseline']#['followupplus', 'baseline', 'baselineplus', 'all'] 

ref_regions = ['gm-cereb','cereb'] #['cereb']#
PVC_flags = ['pvc-' ,'']#['pvc-' ,'']
data_merge_opts = ['baseline', 'baselineplus', 'followupplus', 'all']  #['followupplus', 'baseline', 'baselineplus', 'all'] 
#include_biomarker_list = [['amyloid','flow'],['flow'],['amyloid']]#[['flow'],['amyloid'],['flow','amyloid']]
params = ['amyloid','flow']
path_cmmt=''
label_list = ['m1 PVC','m2 PVC','m1 no PVC','m2 no PVC']
# declaring magnitude of repetition
K = len(region_names)
 
# using list comprehension
# repeat elements K times
label_list_box = [ele for ele in label_list for i in range(K)]

label_list_diff = ['m1-m2 PVC','m1-m2 no PVC']
# declaring magnitude of repetition
#K = len(region_names)
 
# using list comprehension
# repeat elements K times
label_list_box_diff = [ele for ele in label_list_diff for i in range(K)]

for data_merge_opt in data_merge_opts:
    for param in params:
        for ref_region in ref_regions:
            
            out_folder = '/Users/catherinescott/Documents/python_IO_files/SuStaIn_test/SuStaIn_out'
            #datapath = out_folder+'/SUVR_data_merge_out/'+PVC_flag+ref_region
            outpath = out_folder+'/genZscoremodsel_out'+path_cmmt
            
            figures_folder = outpath+'/figures/plot_boxes/'
            if not os.path.exists(figures_folder):
                os.makedirs(figures_folder)
            
            plt.figure()
            plot_data = mean_array[:,0:2,:,ref_regions.index(ref_region),params.index(param),data_merge_opts.index(data_merge_opt)]
            plot_data_reshape = np.concatenate((plot_data[:,0,0],plot_data[:,1,0],plot_data[:,0,1],plot_data[:,1,1]), axis=0)
            
            df = pd.DataFrame({'PVC and mean':label_list_box,data_merge_opt+' '+param+' '+ref_region:plot_data_reshape})
            #df = pd.DataFrame({'PVC and mean':label_list_box,data_merge_opt+' '+param+' '+ref_region:plot_data_reshape.reshape(-1)})
            sns.boxplot(  y=data_merge_opt+' '+param+' '+ref_region, x= 'PVC and mean', data=df,  orient='v' )
            
            plt.savefig(os.path.join(figures_folder,'GMM_mean_'+'_'+ param+'_'+ref_region+'_'+data_merge_opt+'.pdf'))
            
            plt.figure()
            plot_data_reshape_diff = np.concatenate((plot_data[:,0,0]-plot_data[:,1,0],plot_data[:,0,1]-plot_data[:,1,1]), axis=0)
            df_diff= pd.DataFrame({'Difference':label_list_box_diff,data_merge_opt+' '+param+' '+ref_region:plot_data_reshape_diff})
            sns.boxplot(  y=data_merge_opt+' '+param+' '+ref_region, x= 'Difference', data=df_diff,  orient='v' )
            plt.savefig(os.path.join(figures_folder,'GMM_meandiff_'+ param+'_'+ref_region+'_'+data_merge_opt+'.pdf'))