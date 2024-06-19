#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 10:46:30 2024

@author: catherinescott
"""

#this codetakes input csv files and merges them
#input: various csv files containing SUVR data
#function: combines all the available csvs to create 
# 1) all: containing all the available data, 2) merged data using baseline/baselineplu or folllowupplus data
#outputs: csvs above


import os
import glob
import pandas as pd
# import numpy as np
# from scipy.stats import norm
# import matplotlib.pyplot as plt
# from sklearn.mixture import GaussianMixture as GMM
# import matplotlib.patches as mpatches

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
data_merge_opt = 'followupplus' # 'followupplus' 'baseline'

#define paths
out_folder = '/Users/catherinescott/Documents/python_IO_files/SuStaIn_out'
datapath = '/Users/catherinescott/Documents/python_IO_files/input_csv_files/SUVR_spreadsheets/opt_4i2mm/suvr-'+PVC_flag+'nipet-pct-gif-'+ref_region
outpath = out_folder+'/SUVR_data_merge_out'
if not os.path.exists(outpath):
    os.makedirs(outpath)
    
#load in the csv file comtaining the suvr parameters for each subject
#assuming that you want to use all the csv files in the datapath folder
#and early and late csvs are in the same folder
#datacols=['Subject', 'Session','ROI',param+'_srtm', 'R1_srtm']
datacols = ['subject','session']+region_names
all_early_csv_files = glob.glob(os.path.join(datapath, "ses-*0p0to2*.csv"))
all_late_csv_files = glob.glob(os.path.join(datapath, "ses-*40p0to*.csv"))

#read in early files and rename suvr cols
#*note that SUVR files have lowercase 's' for subject, changed to uppercase to match kinetic params
df_orig = pd.concat((pd.read_csv(f, skiprows=0,usecols=datacols) for f in all_early_csv_files), ignore_index=True)
#remove nans
df_early = df_orig.dropna()
#add the name flow to all columns to distinguish from amyloid
df_early = df_early.add_suffix('_flow')
#remove flow from subject column and change to capital S
df_early.rename(columns={'subject_flow':'Subject'}, inplace=True)
df_early.rename(columns={'session_flow':'Session'}, inplace=True)

#read in late files and rename suvr cols
df_orig = pd.concat((pd.read_csv(f, skiprows=0,usecols=datacols) for f in all_late_csv_files), ignore_index=True)
#remove nans
df_late = df_orig.dropna()
df_late = df_late.add_suffix('_amyloid')
df_late.rename(columns={'subject_amyloid':'Subject'}, inplace=True)
df_late.rename(columns={'session_amyloid':'Session'}, inplace=True)


#PART A: creaate a dataframe for all the subjects who will have SUSTAIN modelling ------------------------------------------------------
#merge flow and amyloid markers into single dataframe. Get rid of any that dont have both biomarkers
#need to merge on subject and session to avoid mixing baselines and followups for 1946
df = pd.merge(df_early, df_late, on=['Subject','Session'])
#save a csv file which has all the avilable data in
df.to_csv(os.path.join(outpath, 'all_sustain_raw.csv'))

#options for handling follow up data: option 1- include followup data for subjects where baseline is missing, option 2-only keep baseline data 

if data_merge_opt=='baselineplus': #discards followup for  all those who have baseline data
    duplicated_df = df[df.duplicated(['Subject'], keep=False)]
    idx_to_remove=duplicated_df.index[duplicated_df['Session'] == 'followup'].tolist()
    df.drop(idx_to_remove, inplace=True)
elif data_merge_opt=='baseline': # keeps only the baseine data
    idx_to_remove=df.index[df['Session'] == 'followup'].tolist()
    df.drop(idx_to_remove, inplace=True)    
elif data_merge_opt=='followuppplus': #discards teh baseline for subjects wh have followup
    duplicated_df = df[df.duplicated(['Subject'], keep=False)]
    idx_to_remove=duplicated_df.index[duplicated_df['Session'] == 'baseline'].tolist()
    df.drop(idx_to_remove, inplace=True)
    
#save dataframe of data to be fitted using sustain
df.to_csv(os.path.join(outpath, data_merge_opt+'_sustain_raw.csv'))

# #PART B: create a dataframe contain all the subjects who will be part of the normal database to calculate z-scores-------------------

# #exlusion criteria: cognitive impairment- means removing al YOAD and AVID2 subjects and using cognitive tests for 1946
# #amyloid positivity at baseline or followup

# #estimating Z-scores---------------------------------------------------------

# #generate multiplication factors to get different centiles
# centile_LUT = np.array([[75,0.675],[90,1.282],[95,1.645],[97.5,1.960],[99,2.326]])
# plot_factor = centile_LUT[np.where(centile_LUT[:,0]==plot_centile),1].item()
# cutoff_factor = centile_LUT[np.where(centile_LUT[:,0]==cutoff_centile),1].item()


# image_counter = 0

# #define output dataframe for z scores by patient
# #df_out = df.loc[['Subject']]
# df_out = df[['Subject']].copy()


# #define output dataframe for z_max by region
# df_zmax = pd.DataFrame([region_names,np.ones((len(region_names),1)),np.ones((len(region_names),1)),np.ones((len(region_names),1))*1.960,np.ones((len(region_names),1))]).transpose()
# df_zmax.rename(columns={0: 'Region', 1: 'flow z_max' , 2 : 'amyloid z_max', 3:'amyloid z', 4: 'flow z'}, inplace=True)
# #, columns=['Region','R1 z_max','BP z_max']

# for region in region_names:

        
#     print('Processing '+ region+'--------------------------------------------')    


#     # fit the BPnd data using GMM
#     df_region = df[[region+'_'+param]] #df.loc[(df['Session']=='baseline') & (df['ROI']==region),['Subject',param+'_srtm', 'R1_srtm']]
#     #df_region.dropna(axis=0,inplace=True)
#     X = df_region.to_numpy()
    
    
#     def mix_pdf(x, loc, scale, weights):
#         d = np.zeros_like(x)
#         for mu, sigma, pi in zip(loc, scale, weights):
#             d += pi * norm.pdf(x, loc=mu, scale=sigma)
#         return d
    
#     def single_pdf(x, mu, sigma, weights,component):
#         d = weights[component] * norm.pdf(x, loc=mu[component], scale=sigma[component])
#         return d
    

#     cmap = plt.get_cmap('tab10')
    

#     # fit Gaussian mixture model
#     components = components_to_fit
    
#     gmm = GMM(n_components=components).fit(X)
#     labels = gmm.predict(X)
#     #plt.hist(X, 150)
    

#     #mix = GaussianMixture(n_components=K, random_state=1, max_iter=100).fit(galaxies)
#     pi, mu, sigma = gmm.weights_.flatten(), gmm.means_.flatten(), np.sqrt(gmm.covariances_.flatten())
    
#     grid = np.arange(np.min(X), np.max(X), 0.01)
#         #plot standard histogram (normalised)
#     _ = plt.figure(image_counter)
#     _ = plt.hist(X, bins=100, density=True, alpha=0.2)
#     #plot sum of fitted values
#     _ = plt.plot(grid, mix_pdf(grid, mu, sigma, pi), label='sum', c=cmap(0))
        
#     #plot individual components
#     print('N components ='+str(components))
    
#     #sort by ascending so group 0 is consistantly controls
#     arr1inds = mu.argsort()
#     mu = mu[arr1inds[:]]
#     sigma = sigma[arr1inds[:]]
#     pi = pi[arr1inds[:]]
    
#     # MAKING PLOTS -----------------------------------------------------------
#     for i in range(components):
            
#         _ = plt.plot(grid, single_pdf(grid, mu, sigma, pi, i),'--',c = cmap(i+1), label='group '+str(i))
    
#         print('Group '+str(i))
#         lower_cent = mu[i]-(plot_factor*sigma[i])
#         upper_cent = mu[i]+(plot_factor*sigma[i])
#         print(str(100-plot_centile)+'st percentile='+str(lower_cent)+', '+str(plot_centile)+'th percentile='+str(upper_cent))
#         left, bottom, width, height = (lower_cent, 0, upper_cent-lower_cent, 7)
#         rect=mpatches.Rectangle((left,bottom),width,height, 
#                         #fill=False,
#                         alpha=0.03,
#                        facecolor= cmap(i+1))#,
#                        #linewidth=2, color=cmap(i+1))
#         plt.gca().add_patch(rect)
#         _ = plt.plot([lower_cent,lower_cent],[0,7],':',c = cmap(i+1),alpha=0.3)
#         _ = plt.plot([upper_cent,upper_cent],[0,7],':',c = cmap(i+1),alpha=0.3)
            
#         #plt.scatter(X, [0.01]*X.shape[0], c=labels, cmap='viridis')
    
#     _ = plt.legend(loc='upper right')
#     _ = plt.xlabel(region+' '+param)  
#     _ = plt.ylabel('Probability density')  
#     _ = plt.title('Gaussian mixture model with '+str(components)+
#                       ' components (BIC='+str(round(gmm.bic(X),1))+', AIC='+str(round(gmm.aic(X),1))+')')
#     plt.savefig(os.path.join(outpath,'GMM_'+region+'_'+ str(components)+'component_'+desc+'.pdf'))
#     image_counter = image_counter+1







