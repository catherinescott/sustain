#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


#determining regions and biomarkers to include in modelling (will only be included if there is also sufficient data i.e 2 component GMM)
include_regions = ['frontal','parietal','precuneus','occipital','temporal','insula'] #['frontal','parietal','precuneus','occipital','temporal','insula']#['composite','frontal','parietal','occipital','temporal','insula']#['frontal','parietal','temporal','insula','occipital']
#include_regions = ['frontal', 'parietal','occipital', 'temporal','insula', 'precuneus']


ref_regions = ['gm-cereb','cereb'] #['cereb']#
PVC_flags = ['pvc-', '']#['pvc-' ,'']
data_merge_opts = ['followupplus'] #['followupplus', 'baseline', 'baselineplus', 'all'] 
include_biomarker_list = [['amyloid','flow'],['amyloid'],['flow']]#[['flow'],['amyloid'],['flow','amyloid']]
#include_biomarkers = ['flow'] #['flow', 'amyloid']


for ref_region in ref_regions:
    for PVC_flag in PVC_flags:
        for data_merge_opt in data_merge_opts:
            for include_biomarkers in include_biomarker_list:
            
                print('Running: ref: '+ref_region+', PVC: '+PVC_flag+', datamerge: '+data_merge_opt+' biomarkers:'+' '.join(include_biomarkers))
            
                # test or run
                test_run = 'run' # either 'test' or 'run' determines SuStaIn settings
                
                #remove subjects which were previously put in stage 0?
                remove_zero_subs = 'no'#'yes' #either yes or no
                
                #ref_region = 'cereb' #['cereb', 'gm-cereb']
                #PVC_flag = 'pvc-' #['pvc-' ,'']
                #data_merge_opt = 'followupplus' #['followupplus', 'baseline', 'baselineplus', 'all'] 
                cmmt = PVC_flag+ref_region+'_'+ data_merge_opt
                
                ## data in--------------------------------------------------------------------
                #descriptions to use for input and output data
                
                in_desc = '1946AVID2YOADSUVR_v1'#'1946-srtm-cleanandAVID27'
                if remove_zero_subs=='yes':
                    out_desc = in_desc+'-GMM_'+'_'.join(include_biomarkers)+'_'+test_run+'_'+cmmt+'removezero_v1'
                else:
                    out_desc = in_desc+'-GMM_'+'_'.join(include_biomarkers)+'_'+test_run+'_'+cmmt+'_v1'
                
                
                all_region_names=[]
                #create list to take values from csv files
                for b in include_biomarkers:
                    for rg in include_regions:
                    
                        all_region_names.append(rg+'_'+b+'_z')
                        
                #define paths
                out_folder = '/Users/catherinescott/Documents/python_IO_files/SuStaIn_test/SuStaIn_out'
                datapath = out_folder+'/genZscoremodsel_out/'+PVC_flag+ref_region
                outpath = out_folder+'/run_SuStaIn_GMM/'+PVC_flag+ref_region
                
                #reading in the z-score data
                #datapath = '/Users/catherinescott/Documents/SuStaIn_out'
                csvfile_in = datapath+'/zscore_allregions_'+in_desc+'-'+cmmt+'.csv'
                
                
                #load in data (size M subjects by N biomarkers, data must be z-scored)
                #doesnt skip the header row so that the biomarkers can be ordered according to region_names
                #(header row removed in conversion to numpy)
                
                # if we want to get rid of subjects in stage zero *note that I havent fixed this for the updates i have made to filenames 21/06/2024
                if remove_zero_subs=='yes':
                    df_subjs_to_drop = pd.read_csv('/Users/catherinescott/Documents/SuStaIn_out/SuStaIn_crossvalandanalysis/1946AVID2YOADSUVR_v1-GMM_flow_amyloid_run__v1stagezerosubjs.csv')
                    df_all = pd.read_csv(csvfile_in)
                    cond = df_all['Subject'].isin(df_subjs_to_drop['Subject'])
                    df_all.drop(df_all[cond].index, inplace = True)
                    #update name of csv file to read in
                    csvfile_in = csvfile_in[:len(csvfile_in)-4] + '_removezero' + csvfile_in[len(csvfile_in)-4:]
                    #save new csv
                    df_all.to_csv(csvfile_in)
                    
                df = pd.read_csv(csvfile_in) #, usecols=region_names)[region_names]
                # get relevent columns
                df_cols_z = [s for s in list(df) if "_z" in s]
                available_region_names = list(set(all_region_names).intersection(df_cols_z))
                # removing names not in the orginal list preserves the list order
                region_names = [i for i in all_region_names if i in available_region_names]
                #region_names2 = list(set(df_cols_z).intersection(region_names))
                df = pd.read_csv(csvfile_in, usecols=region_names)[region_names]
                
                data = df.to_numpy()
                #remove nans
                data = data[~np.isnan(data).any(axis=1)]
                # delete row certain subjects as they seem to cause issues in fitting
                if PVC_flag=='pvc-':
                    if ref_region=='cereb':
                        if data_merge_opt=='baseline':
                            data = np.delete(data, 311, 0)
                        elif data_merge_opt=='baselineplus':
                            data = np.delete(data, 439, 0)                            
                        elif data_merge_opt=='followupplus':
                            data = np.delete(data, 439, 0)
                        elif data_merge_opt=='all':
                            data = np.delete(data, 654, 0)                            
                    if ref_region=='gm-cereb':
                        if data_merge_opt=='baseline':                        
                            data = np.delete(data, 18, 0)
                        elif data_merge_opt=='baselineplus':
                            data = np.delete(data, 18, 0)                            
                        elif data_merge_opt=='followupplus':
                            data = np.delete(data, 18, 0)
                        elif data_merge_opt=='all':
                            data = np.delete(data, 18, 0)     
                            
                #output naming and folders
                dataset_name = out_desc
                output_folder = outpath+'/'+dataset_name
                if not os.path.isdir(output_folder):
                    os.makedirs(output_folder)
                
                
                ##set params------------------------------------------------------------------
                z_lims_csv = datapath+'/zmax_allregions_'+in_desc+'-'+cmmt+'.csv' 
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
                    z_vals_region = z_lims_df.loc[:,'Region']
                    #loop through values biomarkers I have
                    for i in range(len(z_vals)):
                        #check whether region is included
                        if any(z_vals_region[i]+'_'+b+'_z' in word for word in region_names):
                            idx = region_names.index(z_vals_region[i]+'_'+b+'_z')
                            z_val_i = str(z_vals[i]).replace('[','').replace(']','')
                            z_val_i = np.fromstring(z_val_i,sep=',')
                            for j in range(len(z_val_i)):
                                Z_vals[idx,j] = z_val_i[j]
                            #idx = idx+1
                
                # Z_vals = np.array([[1,2]]*np.shape(data)[1]) #for testing
                
                # Z_max: the maximum z-score reached at the end of the progression
                # size N biomarkers by 1
                #Z_max = np.percentile(data,95,axis=0).transpose() #for testing
                Z_max = np.zeros([len(region_names),1])
                for b in include_biomarkers:
                    #Z_max = np.append(Z_max,z_lims_df.loc[:,b+' z_max'].to_numpy)
                    col_name = b+' z_max'
                    z_max_vals = z_lims_df.loc[:,col_name]
                    z_vals_region = z_lims_df.loc[:,'Region']
                    for i in range(len(z_max_vals)):
                        if any(z_vals_region[i]+'_'+b+'_z' in word for word in region_names):
                            idx = region_names.index(z_vals_region[i]+'_'+b+'_z')
                            Z_max[idx]=z_max_vals[i]
                    #Z_max.append(z_lims_df.loc[:,b+' z_max'].tolist())
                
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
                sustain_input = pySuStaIn.ZscoreSustain(data, #size M subjects by N biomarkers, data must be z-scored
                                              Z_vals, #size N biomarkers by Z z-scores
                                              Z_max, # size N biomarkers by 0
                                              SuStaInLabels, # size N biomarkers by 1
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
                plt.close('all')
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
                    _=plt.suptitle('ref: '+ref_region+', PVC: '+PVC_flag+', datamerge: '+data_merge_opt+' biomarkers:'+' '.join(include_biomarkers))
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
                
                
                    