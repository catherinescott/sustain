#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 14:46:55 2022

@author: catherinescott
"""
#loop over regions
# for BPnd I can use the GMM directly to generate the z scores
# for R1, for each region read in the relevent csv file generated in mixturemodel.py
# calc mean and standard deviation, then estiamte the z-scores for all subjects 


#VERSION CONTROL:
#v1: copied from gen_zscore_GMM_v3.py to be adapted for SUVR data
#didnt change the version but on 26/1/24 I changed the number of subject per z level to 15
#and added plots


from sklearn.mixture import GaussianMixture as GMM
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import glob
from scipy.stats import norm
from get_subj_status_SUVR import get_subj_status_SUVR
import matplotlib.patches as mpatches

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
desc = '1946AVID2YOADSUVR_v'+version_no+'-'+param #1946-srtm-cleanandAVID27-'+param

#define paths
out_folder = '/Users/catherinescott/Documents/SuStaIn_out'
datapath = '/Users/catherinescott/Documents/SUVR_spreadsheets/opt_10iPSFnoF/csv_to_use'
outpath = out_folder+'/genZscore_out'
if not os.path.exists(outpath):
    os.makedirs(outpath)

#load in the csv file comtaining the suvr parameters for each subject
#assuming that you want to use all the csv files in the datapath folder
#and early and late csvs are in the same folder
#datacols=['Subject', 'Session','ROI',param+'_srtm', 'R1_srtm']
datacols = ['subject']+region_names
all_early_csv_files = glob.glob(os.path.join(datapath, "ses-baseline_0p0to2*.csv"))
all_late_csv_files = glob.glob(os.path.join(datapath, "ses-baseline_40p0to*.csv"))

#read in early files and rename suvr cols
#*note that SUVR files have lowercase 's' for subject, changed to uppercase to match kinetic params
df_orig = pd.concat((pd.read_csv(f, skiprows=0,usecols=datacols) for f in all_early_csv_files), ignore_index=True)
#remove nans
df_early = df_orig.dropna()
#add the name flow to all columns to distinguish from amyloid
df_early = df_early.add_suffix('_flow')
#remove flow from subject column and change to capital S
df_early.rename(columns={'subject_flow':'Subject'}, inplace=True)

#read in late files and rename suvr cols
df_orig = pd.concat((pd.read_csv(f, skiprows=0,usecols=datacols) for f in all_late_csv_files), ignore_index=True)
#remove nans
df_late = df_orig.dropna()
df_late = df_late.add_suffix('_amyloid')
df_late.rename(columns={'subject_amyloid':'Subject'}, inplace=True)

#merge flow and amyloid markers into single dataframe. Get rid of any that dont have both biomarkers
df = pd.merge(df_early, df_late, on='Subject')

#estimating Z-scores---------------------------------------------------------

#generate multiplication factors to get different centiles
centile_LUT = np.array([[75,0.675],[90,1.282],[95,1.645],[97.5,1.960],[99,2.326]])
plot_factor = centile_LUT[np.where(centile_LUT[:,0]==plot_centile),1].item()
cutoff_factor = centile_LUT[np.where(centile_LUT[:,0]==cutoff_centile),1].item()


image_counter = 0

#define output dataframe for z scores by patient
#df_out = df.loc[['Subject']]
df_out = df[['Subject']].copy()


#define output dataframe for z_max by region
df_zmax = pd.DataFrame([region_names,np.ones((len(region_names),1)),np.ones((len(region_names),1)),np.ones((len(region_names),1))*1.960,np.ones((len(region_names),1))]).transpose()
df_zmax.rename(columns={0: 'Region', 1: 'flow z_max' , 2 : 'amyloid z_max', 3:'amyloid z', 4: 'flow z'}, inplace=True)
#, columns=['Region','R1 z_max','BP z_max']

for region in region_names:

        
    print('Processing '+ region+'--------------------------------------------')    


    # fit the BPnd data using GMM
    df_region = df[[region+'_'+param]] #df.loc[(df['Session']=='baseline') & (df['ROI']==region),['Subject',param+'_srtm', 'R1_srtm']]
    #df_region.dropna(axis=0,inplace=True)
    X = df_region.to_numpy()
    
    
    def mix_pdf(x, loc, scale, weights):
        d = np.zeros_like(x)
        for mu, sigma, pi in zip(loc, scale, weights):
            d += pi * norm.pdf(x, loc=mu, scale=sigma)
        return d
    
    def single_pdf(x, mu, sigma, weights,component):
        d = weights[component] * norm.pdf(x, loc=mu[component], scale=sigma[component])
        return d
    

    cmap = plt.get_cmap('tab10')
    

    # fit Gaussian mixture model
    components = components_to_fit
    
    gmm = GMM(n_components=components).fit(X)
    labels = gmm.predict(X)
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
    
    #sort by ascending so group 0 is consistantly controls
    arr1inds = mu.argsort()
    mu = mu[arr1inds[:]]
    sigma = sigma[arr1inds[:]]
    pi = pi[arr1inds[:]]
    
    # MAKING PLOTS -----------------------------------------------------------
    for i in range(components):
            
        _ = plt.plot(grid, single_pdf(grid, mu, sigma, pi, i),'--',c = cmap(i+1), label='group '+str(i))
    
        print('Group '+str(i))
        lower_cent = mu[i]-(plot_factor*sigma[i])
        upper_cent = mu[i]+(plot_factor*sigma[i])
        print(str(100-plot_centile)+'st percentile='+str(lower_cent)+', '+str(plot_centile)+'th percentile='+str(upper_cent))
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
    
    _ = plt.legend(loc='upper right')
    _ = plt.xlabel(region+' '+param)  
    _ = plt.ylabel('Probability density')  
    _ = plt.title('Gaussian mixture model with '+str(components)+
                      ' components (BIC='+str(round(gmm.bic(X),1))+', AIC='+str(round(gmm.aic(X),1))+')')
    plt.savefig(os.path.join(outpath,'GMM_'+region+'_'+ str(components)+'component_'+desc+'.pdf'))
    image_counter = image_counter+1

    # SUBJECT STATUS ---------------------------------------------------------

    #determine whether subjects are controls or not (data is sorted so that group 0 is minimum)
    #use composite regions and param to determine status
    upper_cent = mu[0]+(cutoff_factor*sigma[0])#99th percentile 2.33, 97.5th percentile 1.960
    print('BPnd cut-off used for '+region+' = '+str(upper_cent))       
    #region_df = get_subj_status_SUVR(df, cutoff=upper_cent, region=region, param='amyloid')
    
    if region == 'composite':
        region_df = get_subj_status_SUVR(df, cutoff=upper_cent, region=region, param='amyloid')
        # if it is the composite region then add the subject status to the output dataframe
        df_out['Status'] = region_df.loc[:,'Status']
    
    # Z SCORE FOR AMYLOID----------------------------------------------------
    # calculate z-scores for amyloid based on GMM group 0 mu and sigma (recalc X as previously I removed the NaNs)
    df_region = df[['Subject',region+'_flow',region+'_amyloid']]
    #df_region = df.loc[(df['Session']=='baseline') & (df['ROI']==region),['Subject',param+'_srtm', 'R1_srtm']]
    X = df_region[region+'_amyloid'].to_numpy()

    df_out[region+'_'+param+'_suvr'] = X
    df_out[region+'_'+param+'_z']= (X-mu[0])/sigma[0]
    
    # Z SCORE FOR FLOW------------------------------------------------
    #calculate z score for R1 using the patients who are below the BPnd cutoff (only works if composite is first)
    df_ctl = df_region.drop(df_region[df_out.Status == 'PT'].index) # getting only the controls
    
    # compute the mean and standard deviation of the control population
    Y = df_ctl[[region+'_flow']].to_numpy()
    mean_control = Y.mean()
    std_control = Y.std()
    
    #here I multiply the z score by -1 as blood flow reduces as the disease progresses
    X = df_region[region+'_flow'].to_numpy()
    df_out[region+'_flow_suvr'] = X 
    X_z = -1*((X-mean_control)/std_control)
    df_out[region+'_flow_z']= X_z # -1*((X-mean_control)/std_control)
    
    #set zmax by region and by biomarker 
    #for BPnd use the median value of the second peak to define max
    df_zmax.at[df_zmax.loc[df_zmax['Region'] == region].index[0],'amyloid z_max']=(mu[1]-mu[0])/sigma[0]
    
    #for R1 use 3 standard deviations provded that you have at least 10 subjects in the highest score
    if len(X_z[X_z>3])>15:
        R1_max = 5
        R1_z = [1,2,3]
    elif len(X_z[X_z>2])>15:
        R1_max = 3
        R1_z = [1,2]
    else:
        R1_max = 2
        R1_z= [1]

    df_zmax.at[df_zmax.loc[df_zmax['Region'] == region].index[0],'flow z_max']=R1_max
    df_zmax.at[df_zmax.loc[df_zmax['Region'] == region].index[0],'flow z']=R1_z   
#write out the complete dataframe with all params    
df_out.to_csv(os.path.join(outpath, 'zscore_allregions_'+ str(components)+'component_'+desc+'_co'+str(cutoff_centile)+'th.csv'))
df_zmax.to_csv(os.path.join(outpath, 'zmax_allregions_'+ str(components)+'component_'+desc+'_co'+str(cutoff_centile)+'th.csv'))
     
    #image_counter = image_counter+1
        #plt.show()