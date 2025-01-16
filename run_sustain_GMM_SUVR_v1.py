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
import seaborn as sns
import sklearn.model_selection
import copy


#determining regions and biomarkers to include in modelling (will only be included if there is also sufficient data i.e 2 component GMM)
include_regions = ['frontal','parietal','precuneus','occipital','temporal','insula'] #['frontal','parietal','precuneus','occipital','temporal','insula']#['composite','frontal','parietal','occipital','temporal','insula']#['frontal','parietal','temporal','insula','occipital']
#include_regions = ['frontal', 'parietal','occipital', 'temporal','insula', 'precuneus']


# ref_regions = ['gm-cereb','cereb'] #['cereb']#
# PVC_flags = ['pvc-', '']#['pvc-' ,'']
# data_merge_opts = ['baseline', 'baselineplus'] #['followupplus', 'baseline', 'baselineplus', 'all'] 
# include_biomarker_list = [['amyloid','flow'],['amyloid'],['flow']]#[['flow'],['amyloid'],['flow','amyloid']]

ref_regions = ['gm-cereb']#['gm-cereb','cereb'] #['cereb']#
PVC_flags = ['pvc-']#['pvc-' ,'']
data_merge_opts = ['']#['baseline', 'baselineplus', 'followupplus', 'all']  #['followupplus', 'baseline', 'baselineplus', 'all'] 
include_biomarker_list = [['amyloid']]#[['amyloid','flow'],['flow'],['amyloid']]#[['flow'],['amyloid'],['flow','amyloid']]
z_method = '' # '_SC' or '' for supercontrol derfined z scores or GMM defined respectively
path_cmmt = '_AAIC'#'_noBIC_goldilocks' #'_single' # single indicates that a single z-score level is used. set intersection of GMM as the z-scre level and m2 as the max
cross_val = 'no' # 'yes' will do cross validation, any other response wont


#remove subjects which were previously put in stage 0?
remove_zero_subs = 'no'#'yes' #either yes or no

# ref_regions = ['gm-cereb'] #['cereb']#
# PVC_flags = ['pvc-']#['pvc-' ,'']
# data_merge_opts = ['followupplus']  #['followupplus', 'baseline', 'baselineplus', 'all'] 
# include_biomarker_list = [['amyloid','flow'],['flow'],['amyloid']]#[['flow'],['amyloid'],['flow','amyloid']]
# z_method = '_SC' # '_SC' or '' for supercontrol derfined z scores or GMM defined respectively
# path_cmmt = '' #'

for ref_region in ref_regions:
    for PVC_flag in PVC_flags:
        for data_merge_opt in data_merge_opts:
            for include_biomarkers in include_biomarker_list:
            
                print('Running: ref: '+ref_region+', PVC: '+PVC_flag+', datamerge: '+data_merge_opt+' biomarkers:'+' '.join(include_biomarkers)+', z-score levels: '+path_cmmt)
            
                # test or run
                test_run = 'run' # either 'test' or 'run' determines SuStaIn settings
                

                
                #ref_region = 'cereb' #['cereb', 'gm-cereb']
                #PVC_flag = 'pvc-' #['pvc-' ,'']
                #data_merge_opt = 'followupplus' #['followupplus', 'baseline', 'baselineplus', 'all'] 
                cmmt = PVC_flag+ref_region+'_'+ data_merge_opt
                
                ## data in--------------------------------------------------------------------
                #descriptions to use for input and output data
                
                in_desc = '1946AVID2YOADSUVR_v1'#'1946-srtm-cleanandAVID27'
                if remove_zero_subs=='yes':
                    out_desc = in_desc+'-GMM_'+'_'.join(include_biomarkers)+'_'+test_run+'_'+cmmt+z_method+path_cmmt+'removezero_v1'
                else:
                    out_desc = in_desc+'-GMM_'+'_'.join(include_biomarkers)+'_'+test_run+'_'+cmmt+z_method+path_cmmt+'_v1'

                #define paths
                out_folder = '/Users/catherinescott/Documents/python_IO_files/sustain_test/SuStaIn_out'
                if path_cmmt=='_AAIC':
                    datapath = out_folder+'/genZscoreAAIC_out'+'/'+PVC_flag+ref_region
                else:
                    datapath = out_folder+'/genZscoremodsel_out'+path_cmmt+'/'+PVC_flag+ref_region
                # if remove_zero_subs=='yes':
                #     zero_subs_cmmt='_removezero'
                # else:
                #     zero_subs_cmmt=''
                outpath = out_folder+'/run_SuStaIn_GMM'+path_cmmt+'/'+PVC_flag+ref_region+z_method

                #output naming and folders
                dataset_name = out_desc
                output_folder = outpath+'/'+dataset_name
                if not os.path.isdir(output_folder):
                    os.makedirs(output_folder)                
                
                all_region_names=[]
                #create list to take values from csv files
                for b in include_biomarkers:
                    for rg in include_regions:
                    
                        all_region_names.append(rg+'_'+b+'_z')
                        
                
                #reading in the z-score data
                #datapath = '/Users/catherinescott/Documents/SuStaIn_out'
                if path_cmmt=='_AAIC':
                    csvfile_in = datapath+'/zscore_allregions_2component_'+in_desc+'-amyloid_co97.5th'+'.csv'
                else:
                    csvfile_in = datapath+'/zscore_allregions_'+in_desc+'-'+cmmt+z_method+'.csv'
                
                
                #load in data (size M subjects by N biomarkers, data must be z-scored)
                #doesnt skip the header row so that the biomarkers can be ordered according to region_names
                #(header row removed in conversion to numpy)
                
                # if we want to get rid of subjects in stage zero (uses single subtype s=0 results)
                if remove_zero_subs=='yes':
                    df_subjs_to_drop = pd.read_csv(output_folder.replace('removezero','')+'stagezerosubjs_s0_'+ref_region+'_'+PVC_flag+data_merge_opt+z_method+path_cmmt+'.csv')
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
                subjs= pd.read_csv(csvfile_in, usecols=['Subject'])
                if path_cmmt=='_AAIC':
                    status= pd.read_csv(csvfile_in, usecols=['Status']) 
                    status.rename(columns={'Status':'status'},inplace=True)
                else:
                    status= pd.read_csv(csvfile_in, usecols=['status'])
                
                data = df.to_numpy()
                #remove nans
                #data = data[~np.isnan(data).any(axis=1)]
                nonNaN_subjects = ~np.isnan(data).any(axis=1)
                data = data[nonNaN_subjects]
                subjs = subjs[nonNaN_subjects].reset_index(drop=True)
                status = status[nonNaN_subjects].reset_index(drop=True)     
                
                # # delete row for 01-034 as they seem to cause issues in fitting based on amyloid
                # if PVC_flag=='pvc-':
                #     if ref_region=='cereb':
                #         if data_merge_opt=='baseline':
                #             data = np.delete(data, 311, 0)
                #             subjs.drop([311],inplace=True)
                #             status.drop([311],inplace=True) 

                #         elif data_merge_opt=='baselineplus':
                #             data = np.delete(data, 439, 0)    
                #             subjs.drop([439],inplace=True)
                #             status.drop([439],inplace=True)
                            
                #         elif data_merge_opt=='followupplus':
                #             data = np.delete(data, 439, 0)
                #             subjs.drop([439],inplace=True)
                #             status.drop([439],inplace=True)
                #         elif data_merge_opt=='all':
                #             data = np.delete(data, 654, 0)      
                #             subjs.drop([654],inplace=True)
                #             status.drop([654],inplace=True)
                #     if ref_region=='gm-cereb':
                #         if data_merge_opt=='baseline':                        
                #             data = np.delete(data, 18, 0)
                #             subjs.drop([18],inplace=True)
                #             status.drop([18],inplace=True)

                #         elif data_merge_opt=='baselineplus':
                #             data = np.delete(data, 18, 0)    
                #             subjs.drop([18],inplace=True)
                #             status.drop([18],inplace=True)
                #         elif data_merge_opt=='followupplus':
                #             data = np.delete(data, 18, 0)
                #             subjs.drop([18],inplace=True)
                #             status.drop([18],inplace=True)
                #         elif data_merge_opt=='all':
                #             data = np.delete(data, 18, 0) 
                #             subjs.drop([18],inplace=True)
                #             status.drop([18],inplace=True)
                          
                        
                
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
                if path_cmmt=='_AAIC':
                    z_lims_csv = datapath+'/zmax_allregions_2component_'+in_desc+'-amyloid_co97.5th'+'.csv'  
                    
                else:   
                    z_lims_csv = datapath+'/zmax_allregions_'+in_desc+'-'+cmmt+z_method+'.csv' 
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
                N_S_max = 4  #max number of subtypes to fit
                if test_run == 'test':
                    
                    print('Running with test params:') 
                    N_startpoints = 10 #25 recommended, 10 for testing
                    N_iterations_MCMC = int(1e4) #int(1e5) or int(1e6) recommended, 1e4 for testing
                    print(str(N_startpoints)+' starting points, '+str(N_iterations_MCMC)+' MCMC iterations')
                elif test_run == 'run': 
                    print('Running with run params:') 
                    N_startpoints = 25 #25 recommended, 10 for testing
                    N_iterations_MCMC = int(1e6) #int(1e5) or int(1e6) recommended, 1e4 for testing
                    print(str(N_startpoints)+' starting points, '+str(N_iterations_MCMC)+' MCMC iterations')
                else:
                    print('Test or run not given, assume test...')
                    N_startpoints = 10 #25 recommended, 10 for testing
                    N_iterations_MCMC = int(1e4) #int(1e5) or int(1e6) recommended, 1e4 for testing
                    print(str(N_startpoints)+' starting points, '+str(N_iterations_MCMC)+' MCMC iterations')
                    
                SuStaInLabels = region_names
                
                if '_goldilocks' not in path_cmmt:
                    print('Thresholding the data at max z=20')
                    # this threshold the max z-score at 20 as when using the default sustain params it will crash otherwise
                    data[data>20]=20.00
                
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
                    #due to using zscore levels not equal to 1,2,3 the positional variance plotting function doesnt have enough colours to represent all the levels
                    #for plotting tell the algorithm that the levels are 1,2,3.
                    sustain_input_plot=copy.deepcopy(sustain_input)
                    Z_vals_plot = np.zeros(sustain_input_plot.Z_vals.shape)
                    for i in range(Z_vals_plot.shape[1]):
                        Z_vals_plot[:,i]=i+1
                    #need to make sure that for any biomarkers which dont have 3 levels, the missing levels are replaced with zeros
                    threshold_array = sustain_input.Z_vals.copy()
                    threshold_array[threshold_array>0.0001]=1.0
                    threshold_array[threshold_array<0.0001]=0.0                    
                    sustain_input_plot.Z_vals=Z_vals_plot*threshold_array

                    _ = plt.figure(2)
                    pySuStaIn.ZscoreSustain._plot_sustain_model(sustain_input_plot,samples_sequence,samples_f,len(data),biomarker_labels=SuStaInLabels)
                    _=plt.suptitle('ref: '+ref_region+', PVC: '+PVC_flag+', datamerge: '+data_merge_opt+' biomarkers:'+' '.join(include_biomarkers))
                    plt.savefig(os.path.join(output_folder,'SuStaIn_output_subtype_'+ str(s)+out_desc+'.pdf'))
                    

                    #analysis of results
                    #pickle_filename_s = pickle_path + '/pickle_files/' + dataset_name + '_subtype' + str(s) + '.pickle'
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
                    stagezerosubs.to_csv(output_folder+'stagezerosubjs_s'+str(s)+'_'+ref_region+'_'+PVC_flag+data_merge_opt+z_method+path_cmmt+'.csv') # need to come back to this to implement in a more general way
                    
                    zdata.ml_subtype.value_counts()
                    
                    
                    biomarkerstring = ' '.join(include_biomarkers)
                    print('Total subjects included in '+biomarkerstring+': '+str(len(zdata)))
                    
                    for subtype in range(0,s+2):
                    
                        print('for '+biomarkerstring+ ' subtype '+str(subtype)+' n subjects ='+str(len(zdata.loc[zdata['ml_subtype']==subtype,'ml_stage'])))
                    
                    #As a sanity check, let's make sure all the "controls" were given assigned to low stages by SuStaIn
                    
                    plt.figure()
                    sns.displot(x='ml_stage',hue='Status',data=zdata,col='ml_subtype')
                    plt.savefig(os.path.join(output_folder,'stagebysubtype_'+out_desc+'_'+str(s)+'.pdf'))
                    
                    
                    #And now, let's plot the subtype probabilities over SuStaIn stages to make sure we don't have any crossover events
                    plt.figure()
                    sns.pointplot(x='ml_stage',y='prob_ml_subtype', # input variables
                                  hue='ml_subtype',                 # "grouping" variable
                                data=zdata[zdata.ml_subtype>0]) # only plot for Subtypes 1 and 2 (not 0)
                    plt.ylim(0,1) 
                    plt.axhline(1/(s+1),ls='--',color='k') # plot a line representing change (0.5 in the case of 2 subtypes)
                    plt.savefig(os.path.join(output_folder,'subtypeprob_'+out_desc+'_'+str(s)+'.pdf'))
                    
                #cross validation
                if cross_val == 'yes':
                    # choose the number of folds - here i've used three for speed but i recommend 10 typically
                    N_folds = 10 #5 changed to 10 on 06/11/2024 at 13:22
                    
                    # I want to make sure that AVID2 and YOAD data get split up in the stratification
                    ss = zdata['Subject'].str.contains(pat='AVID20*', regex=True)
                    dd = zdata['Subject'].str.contains(pat='01-0*', regex=True)
                    zdata.loc[ss,'Status']=2
                    zdata.loc[dd,'Status']=3
                    
                    # generate stratified cross-validation training and test set splits
                    labels = zdata.Status.values
                    cv = sklearn.model_selection.StratifiedKFold(n_splits=N_folds, shuffle=True)
                    cv_it = cv.split(zdata, labels)
                    
                    # SuStaIn currently accepts ragged arrays, which will raise problems in the future.
                    # We'll have to update this in the future, but this will have to do for now
                    test_idxs = []
                    for train, test in cv_it:
                        test_idxs.append(test)
                    
                    # uncertain why the data type needs to be different for the 2 cases 
                    #I think if it needs to be a ragged array it has to be object and otherwuse shoud be int
                    if len(test_idxs[0])!=len(test_idxs[N_folds-1]):#PVC_flag=='pvc-':
                        test_idxs = np.array(test_idxs,dtype='object') #'object','int')
                    else:
                        test_idxs = np.array(test_idxs,dtype='int')
                        
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
                    #print('saving figure as: '+os.path.join(output_folder,'CVIC_'+out_desc+'.pdf'))
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
                        #sustain_input.combine_cross_validated_sequences(i+1, N_folds)
                        
                        N_S_selected = i+1#2
                        
                    #dont need it to replot the original as I've already saved it
                    #pySuStaIn.ZscoreSustain._plot_sustain_model(sustain_input,samples_sequence,samples_f,M,subtype_order=(0,1))
                    #_ = plt.suptitle('SuStaIn output')
                        plt.figure(4+i)
                        sustain_input_plot.combine_cross_validated_sequences(N_S_selected, N_folds)
                        #_ = plt.suptitle('Cross-validated SuStaIn output')
                        plt.savefig(os.path.join(output_folder,'CV_positionalvariance_s'+str(N_S_selected)+out_desc+'.pdf'))
                        
    
    
                
                

                
                
                    