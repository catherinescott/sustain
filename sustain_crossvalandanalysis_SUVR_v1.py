 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 10:59:14 2022

@author: catherinescott
"""

# version control
# v1 copied from v3 of kinetic version

#previous version control:
# v3: not baackwards compatible: changed out_desc to automatically populate based on include_biomarkers
# this means that file names will be different. Initial version imported into GIT

import numpy as np
import pandas as pd
import os
import seaborn as sns
#import nibabel as nib
import pySuStaIn
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import sklearn.model_selection


#determining regions and biomarkers to include in modelling (will only be included if there is also sufficient data i.e 2 component GMM)
include_regions = ['frontal','parietal','precuneus','occipital','temporal','insula'] #['frontal','parietal','precuneus','occipital','temporal','insula']#['composite','frontal','parietal','occipital','temporal','insula']#['frontal','parietal','temporal','insula','occipital']
#include_regions = ['frontal', 'parietal','occipital', 'temporal','insula', 'precuneus']
# test or run
test_run = 'run' # either 'test' or 'run' determines SuStaIn settings
#cmmt=''
#remove subjects which were previously put in stage 0?
remove_zero_subs = 'no' #either yes or no


ref_regions = ['gm-cereb','cereb'] #['cereb']#
PVC_flags = ['', 'pvc-']#['pvc-' ,'']
data_merge_opts = ['baseline', 'baselineplus'] #['followupplus', 'baseline', 'baselineplus', 'all'] 
include_biomarker_list = [['amyloid','flow'],['amyloid'],['flow']]#[['flow'],['amyloid'],['flow','amyloid']]
#include_biomarkers = ['flow'] #['flow', 'amyloid']


for ref_region in ref_regions:
    for PVC_flag in PVC_flags:
        for data_merge_opt in data_merge_opts:
            for include_biomarkers in include_biomarker_list:

                
                print('Running: ref: '+ref_region+', PVC: '+PVC_flag+', datamerge: '+data_merge_opt+' biomarkers:'+' '.join(include_biomarkers))
                cmmt = PVC_flag+ref_region+'_'+ data_merge_opt

#include_biomarkers = ['amyloid']#['flow','amyloid']#['R1', 'BPnd']
                
                in_desc = '1946AVID2YOADSUVR_v1'  #'1946clean_AVID2_v3'
                
                if remove_zero_subs=='yes':
                    out_desc = in_desc+'-GMM_'+'_'.join(include_biomarkers)+'_'+test_run+'_'+cmmt+'removezero_v1'
                else:
                    out_desc = in_desc+'-GMM_'+'_'.join(include_biomarkers)+'_'+test_run+'_'+cmmt+'_v1'
                    
                dataset_name = out_desc
                
                #this is the folder where it will save the results
                out_path = '/Users/catherinescott/Documents/Python_IO_files/SuStaIn_test/SuStaIn_out/'
                output_folder = os.path.join(out_path,'SuStaIn_crossvalandanalysis', PVC_flag+ref_region,dataset_name)
                
                if not os.path.isdir(output_folder):
                    os.makedirs(output_folder)  
    
                #this is the folder where it reads the pickle files from
                pickle_path = os.path.join(out_path,'run_SuStaIn_GMM',PVC_flag+ref_region,out_desc)
                pickle_path_test = os.path.join(out_path,'run_SuStaIn_GMM',PVC_flag+ref_region,out_desc+'test')


                #this where it reads the rest of the data from
                datapath = out_path+'/genZscoremodsel_out/'+PVC_flag+ref_region
                csvfile_in = datapath+'/zscore_allregions_'+in_desc+'-'+PVC_flag+ref_region+'_'+ data_merge_opt+'.csv'
                if remove_zero_subs=='yes':
                    csvfile_in = csvfile_in[:len(csvfile_in)-4] + '_removezero' + csvfile_in[len(csvfile_in)-4:]

# #determining regions and biomarkers to include in modelling
# include_regions = ['frontal','parietal','precuneus','occipital','temporal','insula'] #['frontal','parietal','temporal','insula','occipital']

                all_region_names=[]
                #create list to take values from csv files
                for b in include_biomarkers:
                    for rg in include_regions:
                    
                        all_region_names.append(rg+'_'+b+'_z')

# The SuStaIn output has everything we need. We'll use it to populate our dataframe.

#load in data (size M subjects by N biomarkers, data must be z-scored)
#doesnt skip the header row so that the biomarkers can be ordered according to region_names
#(header row removed in conversion to numpy)
# if we want to get rid of subjects in stage zero
                df_z = pd.read_csv(csvfile_in)
                # get relevent columns
                df_cols_z = [s for s in list(df_z) if "_z" in s]
                available_region_names = list(set(all_region_names).intersection(df_cols_z))
                # removing names not in the orginal list preserves the list order
                region_names = [i for i in all_region_names if i in available_region_names]
                
                df = pd.read_csv(csvfile_in, usecols=region_names)[region_names]
                subjs= pd.read_csv(csvfile_in, usecols=['Subject'])
                status= pd.read_csv(csvfile_in, usecols=['status'])
                
                data = df.to_numpy()
                #remove nans
                nonNaN_subjects = ~np.isnan(data).any(axis=1)
                data = data[nonNaN_subjects]
                subjs = subjs[nonNaN_subjects].reset_index(drop=True)
                status = status[nonNaN_subjects].reset_index(drop=True)

                # delete row for 01-034 as they seem to cause issues in fitting based on amyloid
                if PVC_flag=='pvc-':
                    if ref_region=='cereb':
                        if data_merge_opt=='baseline':
                            data = np.delete(data, 311, 0)
                            subjs.drop([311],inplace=True)
                            status.drop([311],inplace=True) 

                        elif data_merge_opt=='baselineplus':
                            data = np.delete(data, 439, 0)    
                            subjs.drop([439],inplace=True)
                            status.drop([439],inplace=True)
                            
                        elif data_merge_opt=='followupplus':
                            data = np.delete(data, 439, 0)
                            subjs.drop([439],inplace=True)
                            status.drop([439],inplace=True)
                        elif data_merge_opt=='all':
                            data = np.delete(data, 654, 0)      
                            subjs.drop([654],inplace=True)
                            status.drop([654],inplace=True)
                    if ref_region=='gm-cereb':
                        if data_merge_opt=='baseline':                        
                            data = np.delete(data, 18, 0)
                            subjs.drop([18],inplace=True)
                            status.drop([18],inplace=True)

                        elif data_merge_opt=='baselineplus':
                            data = np.delete(data, 18, 0)    
                            subjs.drop([18],inplace=True)
                            status.drop([18],inplace=True)
                        elif data_merge_opt=='followupplus':
                            data = np.delete(data, 18, 0)
                            subjs.drop([18],inplace=True)
                            status.drop([18],inplace=True)
                        elif data_merge_opt=='all':
                            data = np.delete(data, 18, 0) 
                            subjs.drop([18],inplace=True)
                            status.drop([18],inplace=True)
                
                subjs.reset_index(inplace=True)
                status.reset_index(inplace=True)
                zdata = pd.DataFrame(data,columns= list(df.columns))
                #subj_IDs = df.loc[:,]
                zdata['Subject'] = subjs['Subject']
                zdata['Status'] = status['status']
                
                #replace status ctl vs label as 0 and 1
                zdata.replace('CTL',0,inplace=True)
                zdata.replace('PT',1,inplace=True)

                ##set params------------------------------------------------------------------
                z_lims_csv = datapath+'/zmax_allregions_'+in_desc+'-'+PVC_flag+ref_region+'_'+ data_merge_opt+'.csv' 
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
                
                sustain_input = pySuStaIn.ZscoreSustain(data,
                                              Z_vals,
                                              Z_max,
                                              SuStaInLabels,
                                              N_startpoints,
                                              N_S_max, 
                                              N_iterations_MCMC, 
                                              pickle_path_test, 
                                              dataset_name, 
                                              False)
                
                print('got to here')
                #---------------------------------------------------------------------

                M = len(zdata) 
                
                #set the number of subtypes (s = n_subtypes-1)
                s=N_S_max-1
                
                pickle_filename_s = pickle_path_test + '/pickle_files/' + dataset_name + '_subtype' + str(s) + '.pickle'
                pk = pd.read_pickle(pickle_filename_s)
                
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
                
                #can use this to re-run and exclude
                stagezerosubs = zdata.loc[zdata.ml_stage==0,'Subject']
                #stagezerosubs.to_csv(output_folder+'stagezerosubjs.csv')
                
                zdata.ml_subtype.value_counts()
                
                
                biomarkerstring = ' '.join(include_biomarkers)
                print('Total subjects included in '+biomarkerstring+': '+str(len(zdata)))
                
                for subtype in range(0,s+2):
                
                    print('for '+biomarkerstring+ ' subtype '+str(subtype)+' n subjects ='+str(len(zdata.loc[zdata['ml_subtype']==subtype,'ml_stage'])))
                
                #As a sanity check, let's make sure all the "controls" were given assigned to low stages by SuStaIn
                
                plt.figure()
                sns.displot(x='ml_stage',hue='Status',data=zdata,col='ml_subtype')
                plt.savefig(os.path.join(output_folder,'stagebysubtype_'+out_desc+'.pdf'))
                
                
                #And now, let's plot the subtype probabilities over SuStaIn stages to make sure we don't have any crossover events
                plt.figure()
                sns.pointplot(x='ml_stage',y='prob_ml_subtype', # input variables
                              hue='ml_subtype',                 # "grouping" variable
                            data=zdata[zdata.ml_subtype>0]) # only plot for Subtypes 1 and 2 (not 0)
                plt.ylim(0,1) 
                plt.axhline(1/N_S_max,ls='--',color='k') # plot a line representing change (0.5 in the case of 2 subtypes)
                plt.savefig(os.path.join(output_folder,'subtypeprob_'+out_desc+'.pdf'))


                #cross validation
                
                # choose the number of folds - here i've used three for speed but i recommend 10 typically
                N_folds = 5
                
                # generate stratified cross-validation training and test set splits
                labels = zdata.Status.values
                cv = sklearn.model_selection.StratifiedKFold(n_splits=N_folds, shuffle=True)
                cv_it = cv.split(zdata, labels)
                
                # SuStaIn currently accepts ragged arrays, which will raise problems in the future.
                # We'll have to update this in the future, but this will have to do for now
                test_idxs = []
                for train, test in cv_it:
                    test_idxs.append(test)
                test_idxs = np.array(test_idxs,dtype='int') #'object')
                
                # perform cross-validation and output the cross-validation information criterion and
                # log-likelihood on the test set for each subtypes model and fold combination
                CVIC, loglike_matrix     = sustain_input.cross_validate_sustain_model(test_idxs)
                
                
                # go through each subtypes model and plot the log-likelihood on the test set and the CVIC
                print("CVIC for each subtype model: " + str(CVIC))
                print("Average test set log-likelihood for each subtype model: " + str(np.mean(loglike_matrix, 0)))
                
                plt.figure()    
                plt.plot(np.arange(N_S_max,dtype=int),CVIC)
                plt.xticks(np.arange(N_S_max,dtype=int))
                plt.ylabel('CVIC')  
                plt.xlabel('Subtypes model') 
                plt.title('CVIC')
                plt.savefig(os.path.join(output_folder,'CVIC_'+out_desc+'.pdf'))
                
                
                plt.figure()
                df_loglike = pd.DataFrame(data = loglike_matrix, columns = ["s_" + str(i) for i in range(sustain_input.N_S_max)])
                df_loglike.boxplot(grid=False)
                plt.ylabel('Log likelihood')  
                plt.xlabel('Subtypes model') 
                plt.title('Test set log-likelihood across folds')
                plt.savefig(os.path.join(output_folder,'LL_'+out_desc+'.pdf'))
                
                #Another useful output of the cross-validation that you can look at are positional variance diagrams averaged across cross-validation folds. These give you an idea of the variability in the progression patterns across different training datasets
                #this part estimates cross-validated positional variance diagrams
                for i in range(N_S_max):
                    sustain_input.combine_cross_validated_sequences(i+1, N_folds)
                    
                    N_S_selected = i+1#2
                    
                #dont need it to replot the original as I've already saved it
                #pySuStaIn.ZscoreSustain._plot_sustain_model(sustain_input,samples_sequence,samples_f,M,subtype_order=(0,1))
                #_ = plt.suptitle('SuStaIn output')
                    plt.figure(4+i)
                    sustain_input.combine_cross_validated_sequences(N_S_selected, N_folds)
                    _ = plt.suptitle('Cross-validated SuStaIn output')
                    plt.savefig(os.path.join(output_folder,'CV_positionalvariance_s'+str(N_S_selected)+out_desc+'.pdf'))
                    
    
    