#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 14:48:58 2024

@author: catherinescott
"""

import os
import glob
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from get_subj_status_SUVR import get_subj_status_SUVR

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

# parameters to set/test:
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

#this array holds the mean and standard deviation of the fits for group1 and 2
mean_array = np.zeros(shape=(len(region_names), 4,len(PVC_flags),len(ref_regions),len(params), len(data_merge_opts)))
mean_array[:] = np.nan

# step 1) read in relevent csv's----------------------------------------------------------------

image_counter = 0
path_cmmt = '' #'single' # single indicates that a single z-score level is used. set intersection of GMM as the z-scre level and m2 as the max

for ref_region in ref_regions:
    for PVC_flag in PVC_flags:
        for data_merge_opt in data_merge_opts:

            #input/output-
            #description to add to outputs
            version_no = '1'
            #ref_region = 'gm-cereb' # options: 'cereb', 'gm-cereb'
            #PVC_flag = '' # options: 'pvc-', ''
            
            #data_merge_opt = 'followupplus' # options: 'baseline' 'baselineplus' 'followupplus' 'all'
            desc = '1946AVID2YOADSUVR_v'+version_no+'-'+PVC_flag+ref_region+'_'+ data_merge_opt
            
            #define paths
            out_folder = '/Users/catherinescott/Documents/python_IO_files/SuStaIn_test/SuStaIn_out'
            datapath = out_folder+'/SUVR_data_merge_out/'+PVC_flag+ref_region
            outpath = out_folder+'/genZscoremodsel_out'+path_cmmt+'/'+PVC_flag+ref_region
            WMH_path = '/Users/catherinescott/Documents/python_IO_files/input_csv_files/WMH'
            supercontrol_path = '/Users/catherinescott/Documents/python_IO_files/input_csv_files/supercontrol'
            
            if not os.path.exists(outpath):
                os.makedirs(outpath)
                
            #load in all data from csv
            df = pd.read_csv(os.path.join(datapath, desc+'_sustain_raw.csv'))
            #load in supercontrols from csv
            df_supercontrol_ID = pd.read_csv(os.path.join(supercontrol_path,'stable.csv')).rename(columns={"subject":"Subject"}).astype(str)

            df_sc = df.merge(df_supercontrol_ID,on='Subject', how='inner')
            
            # step 2) loop over regions and fit GMM with 1 or 2 components. Use the BIC to determine the better model
            
            #generate multiplication factors to get different centiles
            centile_LUT = np.array([[75,0.675],[90,1.282],[95,1.645],[97.5,1.960],[99,2.326]])
            plot_factor = centile_LUT[np.where(centile_LUT[:,0]==plot_centile),1].item()
            cutoff_factor = centile_LUT[np.where(centile_LUT[:,0]==cutoff_centile),1].item()
            CTL_cutoff = 2
            
            
            
            
            #define output dataframe for z scores by patient
            #df_out = df.loc[['Subject']]
            df_out = df[['Subject']].copy()
            
            
            #define output dataframe for z_max by region
            df_zmax = pd.DataFrame([region_names,np.ones((len(region_names),1)),np.ones((len(region_names),1)),np.ones((len(region_names),1))*1.960,np.ones((len(region_names),1))]).transpose()
            df_zmax.rename(columns={0: 'Region', 1: 'flow z_max' , 2 : 'amyloid z_max', 3:'amyloid z', 4: 'flow z'}, inplace=True)
            #, columns=['Region','R1 z_max','BP z_max']
            
            for region in region_names:
            
            
                print('Processing '+ region+'************************************')    
            
                
                
                for param in params:
                    
                    print(param+'--------------------------------------------')
                    
                    #get region data
                    df_region = df[[region+'_'+param]] 
                    X = df_region.to_numpy()
                    #also for supercontrols
                    df_region_sc = df_sc[[region+'_'+param]] 
                    X_sc = df_region_sc.to_numpy()                    
            
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
                    components_to_fit = [1]#[1,2]
                    BIC = []
                    
                    for components in components_to_fit:
                
                    
                        gmm = GMM(n_components=components).fit(X)
                        labels = gmm.predict(X)
                        #plt.hist(X, 150)
                        
                    
                        #mix = GaussianMixture(n_components=K, random_state=1, max_iter=100).fit(galaxies)
                        pi, mu, sigma = gmm.weights_.flatten(), gmm.means_.flatten(), np.sqrt(gmm.covariances_.flatten())
                        
                        grid = np.arange(np.min(X), np.max(X), 0.01)
                            #plot standard histogram (normalised)
                        _ = plt.figure(image_counter)
                        _ = plt.hist(X_sc, bins=round(len(X_sc)/2), density=True, alpha=0.2)
                        #plot sum of fitted values
                        #_ = plt.plot(grid, mix_pdf(grid, mu, sigma, pi), label='sum', c=cmap(0))
                            
                        #plot individual components
                        print('N components ='+str(components))
                        
                        #sort by ascending for amyloid and decending for flow so group 0 is consistantly controls
                        if param=='amyloid':
                            arr1inds = mu.argsort()
                            mu = mu[arr1inds[:]]
                            sigma = sigma[arr1inds[:]]
                            pi = pi[arr1inds[:]]
                        else:
                            mu_flip = mu*(-1)
                            arr1inds = mu_flip.argsort()
                            mu = mu[arr1inds[:]]
                            sigma = sigma[arr1inds[:]]
                            pi = pi[arr1inds[:]]     
                            
                        #fit supercontrols (always use 1 component)
                        gmm_sc = GMM(n_components=1).fit(X_sc)
                        labels_sc = gmm_sc.predict(X_sc)
                        pi_sc, mu_sc, sigma_sc = gmm_sc.weights_.flatten(), gmm_sc.means_.flatten(), np.sqrt(gmm_sc.covariances_.flatten())
                        #calculate the t-test to compare means https://ethanweed.github.io/pythonbook/05.02-ttest.html
                        #F test to compare vairance
                        
                        #add what it would look like usng the supercontrol fit
                        _ = plt.plot(grid, single_pdf(grid, mu_sc, sigma_sc, pi_sc, 0),'--',c = cmap(3), label='group SC')
                        _ = plt.legend(loc='upper right')
                        _ = plt.xlabel(region+' '+param+', n='+str(len(X_sc)))  
                        _ = plt.ylabel('Probability density')  
                        _ = plt.title('GMM '+str(components)+
                                          ' comp '+PVC_flag+ref_region+'_'+ data_merge_opt+'(BIC='+str(round(gmm.bic(X),1))+', AIC='+str(round(gmm.aic(X),1))+')')                        

                        plt.savefig(os.path.join(outpath,'GMM_'+region+'_'+ param+'_'+str(components)+'component_'+desc+'_SC.pdf'))
                        image_counter = image_counter+1
                        
#                         # find the intersection of the gaussians
#                         if components==2:
#                             x1,x2 = solve_gaussians(mu[0], sigma[0], pi[0],mu[1],sigma[1],pi[1])
#                             #find the correct intersection
#                             if x1>mu[0] and x1<mu[1]:
#                                 x_intersect = x1
#                             else:
#                                 x_intersect = x2
#                             y_intersect =single_pdf(x_intersect, mu, sigma, pi, 0)# 1.0/np.sqrt(2*np.pi*sigma[0]**2) * np.exp(-((x_intersect-mu[0])**2)/(2*sigma[0]**2))
# #mlab.normpdf(x_intersect,mu[0],sigma[0])
                        
#                         # MAKING PLOTS -----------------------------------------------------------
#                         #i = components-1
#                         for i in range(components):
                                
#                             _ = plt.plot(grid, single_pdf(grid, mu, sigma, pi, i),'--',c = cmap(i+1), label='group '+str(i))
                        
#                             #print('Group '+str(i))
#                             lower_cent = mu[i]-(plot_factor*sigma[i])
#                             upper_cent = mu[i]+(plot_factor*sigma[i])
#                             #print(str(100-plot_centile)+'st percentile='+str(lower_cent)+', '+str(plot_centile)+'th percentile='+str(upper_cent))
#                             left, bottom, width, height = (lower_cent, 0, upper_cent-lower_cent, 7)
#                             rect=mpatches.Rectangle((left,bottom),width,height, 
#                                             #fill=False,
#                                             alpha=0.03,
#                                            facecolor= cmap(i+1))#,
#                                            #linewidth=2, color=cmap(i+1))
#                             plt.gca().add_patch(rect)
#                             _ = plt.plot([lower_cent,lower_cent],[0,7],':',c = cmap(i+1),alpha=0.3)
#                             _ = plt.plot([upper_cent,upper_cent],[0,7],':',c = cmap(i+1),alpha=0.3)
                            
#                         #plt.scatter(X, [0.01]*X.shape[0], c=labels, cmap='viridis')
#                         #add intersection
#                         if components==2:
#                             _=plt.plot(x_intersect,y_intersect,'ko',mfc='none')
#                             _=plt.plot(mu[0],single_pdf(mu[0], mu, sigma, pi, 0),'x',c = cmap(1),mfc='none')
#                             _=plt.plot(mu[1],single_pdf(mu[1], mu, sigma, pi, 1),'x',c = cmap(2),mfc='none')                            
#                         _ = plt.legend(loc='upper right')
#                         _ = plt.xlabel(region+' '+param)  
#                         _ = plt.ylabel('Probability density')  
#                         _ = plt.title('GMM '+str(components)+
#                                           ' comp '+PVC_flag+ref_region+'_'+ data_merge_opt+'(BIC='+str(round(gmm.bic(X),1))+', AIC='+str(round(gmm.aic(X),1))+')')                        
#                         plt.savefig(os.path.join(outpath,'GMM_'+region+'_'+ param+'_'+str(components)+'component_'+desc+'.pdf'))
                        

#                         image_counter = image_counter+1
                        
#                         BIC.append(gmm.bic(X))
                        
#                         print(str(components)+' comp: BIC='+str(round(gmm.bic(X),1))+', AIC='+str(round(gmm.aic(X),1))+')')
                        
#                     # see whether we will use this region or not
#                     if BIC[0]<=BIC[1] or abs(BIC[1]-BIC[0])<10:
#                         # one component model has lower BIC, dont include this region
#                         print(region+' '+param+' EXCLUDED as 1 gaussian has lower BIC')
#                         df_zmax.at[df_zmax.loc[df_zmax['Region'] == region].index[0],param+' z_max']='NaN'
#                         df_zmax.at[df_zmax.loc[df_zmax['Region'] == region].index[0],param+' z']='NaN'
#                         _ = plt.xlabel('EXCLUDED '+region+' '+param+' (BIC diff= '+str(round(abs(BIC[1]-BIC[0]),1))+')') 
#                         plt.savefig(os.path.join(outpath,'GMM_'+region+'_'+ param+'_'+str(components)+'component_'+desc+'.pdf'))
                        
#                         #add what it would look like usng the supercontrol fit
#                         _ = plt.plot(grid, single_pdf(grid, mu_sc, sigma_sc, pi_sc, 0),'--',c = cmap(i+2), label='group SC')
#                         _ = plt.legend(loc='upper right')
#                         plt.savefig(os.path.join(outpath,'GMM_'+region+'_'+ param+'_'+str(components)+'component_'+desc+'_SC.pdf'))
                        
#                     else: # two component model has lower BIC
#                         print(region+' '+param+' INCLUDED as 2 gaussian has lower BIC')
#                         mean_array[region_names.index(region),:,PVC_flags.index(PVC_flag),ref_regions.index(ref_region), params.index(param),data_merge_opts.index(data_merge_opt)] = [mu[0],mu[1],sigma[0],sigma[1]]
#                         _ = plt.xlabel(region+' '+param+' (BIC diff= '+str(round(abs(BIC[1]-BIC[0]),1))+')') 
#                         plt.savefig(os.path.join(outpath,'GMM_'+region+'_'+ param+'_'+str(components)+'component_'+desc+'.pdf'))                        
                        
#                         #add what it would look like usng the supercontrol fit
#                         _ = plt.plot(grid, single_pdf(grid, mu_sc, sigma_sc, pi_sc, 0),'--',c = cmap(i+2), label='group SC')
#                         _ = plt.legend(loc='upper right')
#                         plt.savefig(os.path.join(outpath,'GMM_'+region+'_'+ param+'_'+str(components)+'component_'+desc+'_SC.pdf'))
                            
#             # step 3) if the region has 2 components the calculate the z-score for all of the subjects
            
#                         # calculate z-scores based on GMM group 0 mu and sigma (recalc X as previously I removed the NaNs)
#                         df_region = df[['Subject',region+'_'+param]]
#                         #df_region = df.loc[(df['Session']=='baseline') & (df['ROI']==region),['Subject',param+'_srtm', 'R1_srtm']]
#                         X = df_region[region+'_'+param].to_numpy()
                    
#                         df_out[region+'_'+param+'_suvr'] = X
#                         if param == 'flow':
#                             X_z = -1*(X-mu[0])/sigma[0]
#                         else:
#                             X_z = 1*(X-mu[0])/sigma[0]
#                         df_out[region+'_'+param+'_z']= X_z #(X-mu[0])/sigma[0]
                        

                        
            
#             # step 4) determine the number of z-score levels based on the number of subjects at each level
#                         # use 3 standard deviations provded that you have at least 10 subjects in the highest score
#                         print('len(X_z[X_z>3]):'+str(len(X_z[X_z>3])))
#                         print('len(X_z[X_z>2]):'+str(len(X_z[X_z>2])))
#                         print('len(X_z[X_z>1]):'+str(len(X_z[X_z>1]))) 
#                         n_subjects_per_z = 20
                        
#                         if path_cmmt == 'single':
#                             print('using single cut off of intersection')
#                             #when path_cmmt = 'single'
#                             # set cutoffs at intersection and max at mean of second gaussian
#                             #calculate zscore for this
#                             if param == 'flow':
#                                 R1_z = -1*(x_intersect-mu[0])/sigma[0]
#                                 R1_max = -1*(mu[1]-mu[0])/sigma[0]
                                
#                             else:
#                                 R1_z = 1*(x_intersect-mu[0])/sigma[0]
#                                 R1_max = 1*(mu[1]-mu[0])/sigma[0]
#                         else:
                            
#                             print('Using cut offs at 1 2 and 3 std dev')
#                         # for regular use
#                             if len(X_z[X_z>3])>n_subjects_per_z:
#                                 R1_max = 5
#                                 R1_z = [1,2,3]
#                             elif len(X_z[X_z>2])>n_subjects_per_z:
#                                 R1_max = 3
#                                 R1_z = [1,2]
#                             else:
#                                 R1_max = 2
#                                 R1_z= [1]      
                        

#                         print('mu SC: '+str(mu_sc)+', sigma SC: '+str(sigma_sc))
#                         print('mu[0]: '+str(mu[0])+', sigma[0]: '+str(sigma[0])+', x_intersect: '+str(x_intersect))                        
#                         print('mu[1]: '+str(mu[1])+', sigma[1]: '+str(sigma[1])+', x_intersect: '+str(x_intersect))                        
                        
#                         df_zmax.at[df_zmax.loc[df_zmax['Region'] == region].index[0],param+' z_max']=R1_max
#                         df_zmax.at[df_zmax.loc[df_zmax['Region'] == region].index[0],param+' z']=R1_z   
            
#             # step 5) SUBJECT STATUS ---------------------------------------------------------

#             #determine whether subjects are controls or not (data is sorted so that group 0 is minimum)
#             #use composite regions and param to determine status
            
#             #for param in params:
#                #find all relevent z score columns for the parameter in question
#             check_cols = [x for x in df_out.columns if '_z' in x]
#             #determine whether any of the z-scores are >2, will put TRUE if so
#             df_out["PET_PT_status"] = (df_out[check_cols]>=CTL_cutoff).any(axis="columns")
                
            
#             #load in WMH data
#             df_WMH = pd.read_csv(os.path.join(WMH_path, 'Longitudinal_P1-P2_WMHV_dataset.csv'),
#                                  usecols=['subject','p1_bamos_wmhv_long','p2_bamos_wmhv_long'])
            
#             check_cols = [x for x in df_WMH.columns if 'bamos_wmhv_long' in x]
#             WMH_cutoff = 5
#             df_WMH["WMH_PT_status"] = (df_WMH[check_cols]>=WMH_cutoff).any(axis="columns")
#             df_WMH.drop(check_cols,axis=1,inplace=True)
#             df_WMH.rename(columns={"subject": "Subject"}, inplace=True)
            
#             df_WMH['Subject']=df_WMH['Subject'].astype(str)
#             df_out['Subject']=df_out['Subject'].astype(str)
            
            
#             df_out = pd.merge(df_out, df_WMH, how="left",on=["Subject"])
#             check_cols = [x for x in df_out.columns if 'status' in x]
            
#             df_out['status']=(df_out[check_cols]==False).all(axis="columns")
#             df_out['status'] = df_out['status'].replace([True],'CTL')
#             df_out['status'] = df_out['status'].replace([False],'PT')
#             # upper_cent = mu[0]+(CTL_cutoff *sigma[0])#99th percentile 2.33, 97.5th percentile 1.960
#             # print('BPnd cut-off used for '+region+' = '+str(upper_cent))       
#             # #region_df = get_subj_status_SUVR(df, cutoff=upper_cent, region=region, param='amyloid')
            
#             # if region == 'composite':
#             #     df_out['Status'+'_'+param] = np.where(df_out[region+'_'+param+'_z']<=CTL_cutoff,'CTL','PT')

#                 #region_df = get_subj_status_SUVR(df, cutoff=upper_cent, region=region, param='amyloid')
#                 # if it is the composite region then add the subject status to the output dataframe
#                 #df_out['Status'] = region_df.loc[:,'Status']
                        
                    
                    
            
#             # step 6) save the resulting csv files
            
#             #write out the complete dataframe with all params    
#             df_out.to_csv(os.path.join(outpath, 'zscore_allregions_'+desc+'.csv'))
#             df_zmax.to_csv(os.path.join(outpath, 'zmax_allregions_'+desc+'.csv'))
             
        













