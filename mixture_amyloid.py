#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 11:58:41 2025

@author: catherinescott
"""

#code for running mixture SuStaIn (GMM)
import numpy as np
import pandas as pd
import os
import pySuStaIn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
from pathlib import Path
import seaborn as sns
import sklearn.model_selection
import copy
from scipy.stats import norm
from sklearn.mixture import GaussianMixture as GMM

#determining regions and biomarkers to include in modelling (will only be included if there is also sufficient data i.e 2 component GMM)
include_regions = ['frontal','parietal','precuneus','occipital','temporal','insula'] #['frontal','parietal','precuneus','occipital','temporal','insula']#['composite','frontal','parietal','occipital','temporal','insula']#['frontal','parietal','temporal','insula','occipital']

plot_centile = 97.5
cutoff_centile = 97.5

ref_regions = ['gm-cereb']#['gm-cereb','cereb'] #['cereb']#
PVC_flags = ['']#['pvc-' ,'']
data_merge_opts = ['followupplus']#['baseline', 'baselineplus', 'followupplus', 'all']  #['followupplus', 'baseline', 'baselineplus', 'all'] 
params = [['amyloid']]#[['amyloid','flow'],['flow'],['amyloid']]#[['flow'],['amyloid'],['flow','amyloid']]
#z_method = '_SC' # '_SC' or '' for supercontrol derfined z scores or GMM defined respectively
path_cmmt = '' #'_single' # single indicates that a single z-score level is used. set intersection of GMM as the z-scre level and m2 as the max
cross_val = 'no' # 'yes' will do cross validation, any other response wont
#remove subjects which were previously put in stage 0?
remove_zero_subs = 'no'#'yes' #either yes or no

image_counter = 0

# function to get the interceotion between gaussians
def solve_gaussians(m1, s1,w1, m2, s2,w2):
    #coefficients of quadratic equation ax^2 + bx + c = 0
    # a = (s1**2.0) - (s2**2.0)
    # b = 2 * (m1 * s2**2.0 - m2 * s1**2.0)
    # c = m2**2.0 * s1**2.0 - m1**2.0 * s2**2.0 - 2 * s1**2.0 * s2**2.0 * np.log(s1/s2)
    
    a = 1/(2*s1**2) - 1/(2*s2**2)
    b = m2/(s2**2) - m1/(s1**2)
    #c = m1**2 /(2*s1**2) - m2**2 / (2*s2**2) - np.log(s2/s1)
    c = m1**2 /(2*s1**2) - m2**2 / (2*s2**2) - np.log((s2*w1)/(s1*w2))

    x1 = (-b + np.sqrt(b**2.0 - 4.0 * a * c)) / (2.0 * a)
    x2 = (-b - np.sqrt(b**2.0 - 4.0 * a * c)) / (2.0 * a)
    
    # x1 = (s1*s2*np.sqrt((-2*np.log(s1/s2)*s2**2)+2*s1**2*np.log(s1/s2)+m2**2-2*m1*m2+m1**2)+m1*s2**2-m2*s1**2)/(s2**2-s1**2)
    # x2 = -(s1*s2*np.sqrt((-2*np.log(s1/s2)*s2**2)+2*s1**2*np.log(s1/s2)+m2**2-2*m1*m2+m1**2)-m1*s2**2+m2*s1**2)/(s2**2-s1**2)
    return x1, x2

for ref_region in ref_regions:
    for PVC_flag in PVC_flags:
        for data_merge_opt in data_merge_opts:
            for param in params:
            
                print('Running: ref: '+ref_region+', PVC: '+PVC_flag+', datamerge: '+data_merge_opt+' , param: '+' '.join(param))
            
                # test or run
                test_run = 'run' # either 'test' or 'run' determines SuStaIn settings
                #comment for outputs
                cmmt = PVC_flag+ref_region+'_'+ data_merge_opt
                
                
                desc = '1946AVID2YOADSUVR_v1-'+PVC_flag+ref_region+'_'+ data_merge_opt
                
                #define paths
                out_folder = '/Users/catherinescott/Documents/python_IO_files/SuStaIn_test/SuStaIn_out'
                datapath = out_folder+'/SUVR_data_merge_out/'+PVC_flag+ref_region
                outpath = out_folder+'/run_SuStaIn_mixture'+path_cmmt+'/'+PVC_flag+ref_region
                figures_path = outpath+'/figures'
                
                if not os.path.exists(outpath):
                    os.makedirs(outpath)
                if not os.path.exists(figures_path):
                    os.makedirs(figures_path)  
                    
                #output naming and folders
                if remove_zero_subs=='yes':
                    out_desc = desc+'-GMM_'+'_'.join(param)+'_'+test_run+'_'+cmmt+path_cmmt+'removezero_v1'
                else:
                    out_desc = desc+'-GMM_'+'_'.join(param)+'_'+test_run+'_'+cmmt+path_cmmt+'_v1'
                
                dataset_name = out_desc
                output_folder = outpath+'/'+dataset_name
                if not os.path.isdir(output_folder):
                    os.makedirs(output_folder)   
                    
                csvfile_in = (os.path.join(datapath, desc+'_sustain_raw.csv'))
                
                # if we want to get rid of subjects in stage zero (uses single subtype s=0 results)
                if remove_zero_subs=='yes':
                    df_subjs_to_drop = pd.read_csv(output_folder.replace('removezero','')+'stagezerosubjs_s0_'+ref_region+'_'+PVC_flag+data_merge_opt+path_cmmt+'.csv')
                    df_all = pd.read_csv(csvfile_in)
                    cond = df_all['Subject'].isin(df_subjs_to_drop['Subject'])
                    df_all.drop(df_all[cond].index, inplace = True)
                    #update name of csv file to read in
                    csvfile_in = csvfile_in[:len(csvfile_in)-4] + '_removezero' + csvfile_in[len(csvfile_in)-4:]
                    #save new csv
                    df_all.to_csv(csvfile_in)
                
                
                all_region_names=[]
                #create list to take values from csv files
                for b in param:
                    for rg in include_regions:
                    
                        all_region_names.append(rg+'_'+b)
                
                df = pd.read_csv(csvfile_in)
                
                df_cols_z = [s for s in list(df) if "_" in s]
                available_region_names = list(set(all_region_names).intersection(df_cols_z))
                # removing names not in the orginal list preserves the list order
                region_names = [i for i in all_region_names if i in available_region_names]  
                
                df = pd.read_csv(csvfile_in, usecols=region_names)[region_names]
                subjs= pd.read_csv(csvfile_in, usecols=['Subject'])
                
                data = df.to_numpy()
                #remove nans
                #data = data[~np.isnan(data).any(axis=1)]
                nonNaN_subjects = ~np.isnan(data).any(axis=1)
                data = data[nonNaN_subjects]
                subjs = subjs[nonNaN_subjects].reset_index(drop=True)
    
                subjs.reset_index(inplace=True)
    
                zdata = pd.DataFrame(data,columns= list(df.columns))
                #subj_IDs = df.loc[:,]
                zdata['Subject'] = subjs['Subject']
                
                centile_LUT = np.array([[75,0.675],[90,1.282],[95,1.645],[97.5,1.960],[99,2.326]])
                plot_factor = centile_LUT[np.where(centile_LUT[:,0]==plot_centile),1].item()
                cutoff_factor = centile_LUT[np.where(centile_LUT[:,0]==cutoff_centile),1].item()
                CTL_cutoff = 2
                
                #define output dataframe for z scores by patient
                #df_out = df.loc[['Subject']]
                df_yes = subjs[['Subject']].copy()
                df_no = subjs[['Subject']].copy()
                
                for region in region_names:               
                    print('Processing '+ region+'************************************')    
                    #for param in params:                      
                    #    print(param+'--------------------------------------------')                        
                        
                    #get region data
                    df_region = df[[region]] 
                    X = df_region.to_numpy()
                    
                    #functions for gaussian mixture modelling
                    def mix_pdf(x, loc, scale, weights):
                        d = np.zeros_like(x)
                        for mu, sigma, pi in zip(loc, scale, weights):
                            d += pi * norm.pdf(x, loc=mu, scale=sigma)
                        return d
                    
                    def single_pdf(x, mu, sigma, weights,component):
                        d = weights[component] * norm.pdf(x, loc=mu[component], scale=sigma[component])
                        return d
                    
                    # fit Gaussian mixture model
                    # 1 component first
                    cmap = plt.get_cmap('tab10')
                    components_to_fit = [1,2]
                    BIC = []
                    
                                        
                    for components in components_to_fit:
                
                    
                        gmm = GMM(n_components=components).fit(X)
                        labels = gmm.predict(X)
                        prob_groups = gmm.predict_proba(X)
                        #plt.hist(X, 150)
                        
                    
                        #mix = GaussianMixture(n_components=K, random_state=1, max_iter=100).fit(galaxies)
                        pi, mu, sigma = gmm.weights_.flatten(), gmm.means_.flatten(), np.sqrt(gmm.covariances_.flatten())
                        
                        grid = np.arange(np.min(X), np.max(X), 0.01)
                            #plot standard histogram (normalised)
                        _ = plt.figure(image_counter)
                        _ = plt.hist(X, bins=100, density=True, alpha=0.2)
                        #plot sum of fitted values
                        _ = plt.plot(grid, mix_pdf(grid, mu, sigma, pi), label='sum', c=cmap(0))
                            
                        #plot individual components
                        print('N components ='+str(components))
                        
                        #sort by ascending for amyloid and decending for flow so group 0 is consistantly controls
                        if param=='flow':
                            mu_flip = mu*(-1)
                            arr1inds = mu_flip.argsort()
                            mu = mu[arr1inds[:]]
                            sigma = sigma[arr1inds[:]]
                            pi = pi[arr1inds[:]]   

                        else:
                            arr1inds = mu.argsort()
                            mu = mu[arr1inds[:]]
                            sigma = sigma[arr1inds[:]]
                            pi = pi[arr1inds[:]]

                        
                        # find the intersection of the gaussians
                        if components==2:
                            
                            if param=='flow':
                                df_yes[region]=prob_groups[:,0]
                                df_no[region]=prob_groups[:,1]
                            else:
                                df_yes[region]=prob_groups[:,1]
                                df_no[region]=prob_groups[:,0]                                
                            
                            
                            x1,x2 = solve_gaussians(mu[0], sigma[0], pi[0],mu[1],sigma[1],pi[1])
                            #find the correct intersection
                            if x1>mu[0] and x1<mu[1]:
                                x_intersect = x1
                            else:
                                x_intersect = x2
                            y_intersect =single_pdf(x_intersect, mu, sigma, pi, 0)# 1.0/np.sqrt(2*np.pi*sigma[0]**2) * np.exp(-((x_intersect-mu[0])**2)/(2*sigma[0]**2))
    #mlab.normpdf(x_intersect,mu[0],sigma[0])
                        
                        # MAKING PLOTS -----------------------------------------------------------
                        #i = components-1
                        for i in range(components):
                                
                            _ = plt.plot(grid, single_pdf(grid, mu, sigma, pi, i),'--',c = cmap(i+1), label='group '+str(i))
                        
                            #print('Group '+str(i))
                            lower_cent = mu[i]-(plot_factor*sigma[i])
                            upper_cent = mu[i]+(plot_factor*sigma[i])
                            #print(str(100-plot_centile)+'st percentile='+str(lower_cent)+', '+str(plot_centile)+'th percentile='+str(upper_cent))
                            left, bottom, width, height = (lower_cent, 0, upper_cent-lower_cent, 7)
                            rect=mpatches.Rectangle((left,bottom),width,height, 
                                            #fill=False,
                                            alpha=0.03,
                                           facecolor= cmap(i+1))#,
                                           #linewidth=2, color=cmap(i+1))
                            plt.gca().add_patch(rect)
                            _ = plt.plot([lower_cent,lower_cent],[0,7],':',c = cmap(i+1),alpha=0.3)
                            _ = plt.plot([upper_cent,upper_cent],[0,7],':',c = cmap(i+1),alpha=0.3)
                            
                        #plt.scatter(X, [0.01]*X.shape[0], c=labels, cmap='viridis')
                        #add intersection
                        if components==2:
                            _=plt.plot(x_intersect,y_intersect,'ko',mfc='none')
                            _=plt.plot(mu[0],single_pdf(mu[0], mu, sigma, pi, 0),'x',c = cmap(1),mfc='none')
                            _=plt.plot(mu[1],single_pdf(mu[1], mu, sigma, pi, 1),'x',c = cmap(2),mfc='none')                            
                        _ = plt.legend(loc='upper right')
                        _ = plt.xlabel(region+' '+' '.join(param))  
                        _ = plt.ylabel('Probability density')  
                        _ = plt.title('GMM '+str(components)+
                                          ' comp '+PVC_flag+ref_region+'_'+ data_merge_opt+'(BIC='+str(round(gmm.bic(X),1))+', AIC='+str(round(gmm.aic(X),1))+')')                        
                        plt.savefig(os.path.join(figures_path,'GMM_'+region+'_'+ '_'.join(param)+'_'+str(components)+'component_'+desc+'.pdf'))
                        
    
                        image_counter = image_counter+1
                        
                        BIC.append(gmm.bic(X))
                        
                        print(str(components)+' comp: BIC='+str(round(gmm.bic(X),1))+', AIC='+str(round(gmm.aic(X),1))+')')
                
                #get subject status- currently using GMM of composite region to define 
                # this is not very sensitive- could use supercontrols instead
                #using the composite region and use 2 components for GMM
                region = 'composite'
                components = 2
                df_status_in = pd.read_csv(csvfile_in, usecols=[region+'_'+param[0]]) 
                X_status = df_region.to_numpy()
                gmm_status = GMM(n_components=components).fit(X_status)
                labels_status = gmm_status.predict(X_status)
                prob_groups_status = gmm_status.predict_proba(X_status)  
                
                if param=='flow':
                    yes_status=prob_groups_status[:,0]
                    no_status=prob_groups_status[:,1]
                else:
                    yes_status=prob_groups_status[:,1]
                    no_status=prob_groups_status[:,0] 
                    
                yes_status[yes_status>0.5]=1
                yes_status[yes_status<0.5]=0
                zdata['Status']=yes_status
                
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
                

                L_yes = df_yes.copy()                    
                L_yes.drop('Subject', axis=1, inplace=True)                    
                L_yes = L_yes.to_numpy()
                
                L_no = df_no.copy()                    
                L_no.drop('Subject', axis=1, inplace=True)                    
                L_no = L_no.to_numpy()
                
                # Run SuStaIn ---------------------------------------------------------------
                sustain_input = pySuStaIn.MixtureSustain(L_yes,L_no, #probability of positive class, dim: number of subjects x number of biomarkers
                                              SuStaInLabels, # size N biomarkers by 1
                                              N_startpoints,
                                              N_S_max, 
                                              N_iterations_MCMC, 
                                              output_folder, 
                                              dataset_name, 
                                              True)
                
                
                    
                    # runs the sustain algorithm with the inputs set in sustain_input above
                samples_sequence,   \
                samples_f,          \
                ml_subtype,         \
                prob_ml_subtype,    \
                ml_stage,           \
                prob_ml_stage,      \
                prob_subtype_stage  = sustain_input.run_sustain_algorithm()
                

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
                    
                    #positional variance diagram plot
                    _ = plt.figure(2)
                    pySuStaIn.MixtureSustain._plot_sustain_model(sustain_input,samples_sequence,samples_f,len(L_yes),biomarker_labels=SuStaInLabels)
                    _=plt.suptitle('ref: '+ref_region+', PVC: '+PVC_flag+', datamerge: '+data_merge_opt+' biomarkers:'+' '.join(param))
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
                    stagezerosubs.to_csv(output_folder+'stagezerosubjs_s'+str(s)+'_'+ref_region+'_'+PVC_flag+data_merge_opt+path_cmmt+'.csv') # need to come back to this to implement in a more general way
                    
                    zdata.ml_subtype.value_counts()
                    
                    biomarkerstring = ' '.join(param)
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
                                  hue='Status',                 # "grouping" variable
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
                        sustain_input.combine_cross_validated_sequences(N_S_selected, N_folds)
                        #_ = plt.suptitle('Cross-validated SuStaIn output')
                        plt.savefig(os.path.join(output_folder,'CV_positionalvariance_s'+str(N_S_selected)+out_desc+'.pdf'))
                        
    
    
                
                

                
                
                    

# def __init__(self,
    #              L_yes,
    #              L_no,
    #              biomarker_labels,
    #              N_startpoints,
    #              N_S_max,
    #              N_iterations_MCMC,
    #              output_folder,
    #              dataset_name,
    #              use_parallel_startpoints,
    #              seed=None):
        
        
        
        # The initializer for the mixture model based events implementation of AbstractSustain
        # Parameters:
        #   L_yes                       - probability of positive class for all subjects across all biomarkers (from mixture modelling)
        #                                 dim: number of subjects x number of biomarkers
        #   L_no                        - probability of negative class for all subjects across all biomarkers (from mixture modelling)
        #                                 dim: number of subjects x number of biomarkers
        #   biomarker_labels            - the names of the biomarkers as a list of strings
        #   N_startpoints               - number of startpoints to use in maximum likelihood step of SuStaIn, typically 25
        #   N_S_max                     - maximum number of subtypes, should be 1 or more
        #   N_iterations_MCMC           - number of MCMC iterations, typically 1e5 or 1e6 but can be lower for debugging
        #   output_folder               - where to save pickle files, etc.
        #   dataset_name                - for naming pickle files
        #   use_parallel_startpoints    - boolean for whether or not to parallelize the maximum likelihood loop
        #   seed                        - random number seed