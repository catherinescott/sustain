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

#v2: updated the regions to include precuneus and changed the order. changed description
# deleted some commented sections
#v3: removed precuneus, updated naming

from sklearn.mixture import GaussianMixture as GMM
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import glob
from scipy.stats import norm
from get_subj_status import get_subj_status
import matplotlib.patches as mpatches

# parameters to set/test:----------------------------------------------------
param = 'BPnd' #'BPnd' #this is the parameter used to define normality
#see data_cols for all params read in
#(assume that we want to generate z-scores for R1 and BPnd (hardcoded))
region_names =['composite','frontal','parietal','occipital','temporal','insula']
components_to_fit = 2
plot_centile = 97.5
cutoff_centile = 97.5


#input/output-----------------------------------------------------------------
#description to add to outputs
version_no = '3'
desc = '1946clean_AVID2_v'+version_no+'-'+param #1946-srtm-cleanandAVID27-'+param

#define paths
out_folder = '/Users/catherinescott/Documents/SuStaIn_out'
datapath = out_folder+'/csvfiles'
outpath = out_folder+'/genZscore_out'
if not os.path.exists(outpath):
    os.makedirs(outpath)

#load in the csv file comtaining the srtm parameters for each subject
#assuming that you want to use all the csv files in the datapath folder
datacols=['Subject', 'Session','ROI',param+'_srtm', 'R1_srtm']
all_csv_files = glob.glob(os.path.join(datapath, "*.csv"))
df_orig = pd.concat((pd.read_csv(f, skiprows=0,usecols=datacols) for f in all_csv_files), ignore_index=True)
#remove nans
df = df_orig.dropna()

#estimating Z-scores---------------------------------------------------------

#generate multiplication factors to get different centiles
centile_LUT = np.array([[75,0.675],[90,1.282],[95,1.645],[97.5,1.960],[99,2.326]])
plot_factor = centile_LUT[np.where(centile_LUT[:,0]==plot_centile),1].item()
cutoff_factor = centile_LUT[np.where(centile_LUT[:,0]==cutoff_centile),1].item()


image_counter = 0

#define output dataframe for z scores by patient
df_out = df.loc[(df['Session']=='baseline') & (df['ROI']=='composite'),['Subject']]
#drop last row because the datapoint is weird
df_out.drop(df_out.tail(1).index, axis=0, inplace=True)

#define output dataframe for z_max by region
df_zmax = pd.DataFrame([region_names,np.ones((len(region_names),1)),np.ones((len(region_names),1)),np.ones((len(region_names),1))*1.960,np.ones((len(region_names),1))]).transpose()
df_zmax.rename(columns={0: 'Region', 1: 'R1 z_max' , 2 : 'BPnd z_max', 3:'BPnd z', 4: 'R1 z'}, inplace=True)
#, columns=['Region','R1 z_max','BP z_max']

for region in region_names:

        
    print('Processing '+ region+'--------------------------------------------')    


    # fit the BPnd data using GMM
    df_region = df.loc[(df['Session']=='baseline') & (df['ROI']==region),['Subject',param+'_srtm', 'R1_srtm']]
    df_region.dropna(axis=0,inplace=True)
    X = df_region[[param+'_srtm']].to_numpy()
    X = X[0:(len(X)-1)] #the last datapoint is weird, need to investigate, remove for now
    

    
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
    
    #plot individual components
    print('N components ='+str(components))
    
    #sort by ascending so group 0 is consistantly controls
    arr1inds = mu.argsort()
    mu = mu[arr1inds[:]]
    sigma = sigma[arr1inds[:]]
    pi = pi[arr1inds[:]]
    
 
    #determine whether subjects are controls or not (data is sorted so that group 0 is minimum)
    upper_cent = mu[0]+(cutoff_factor*sigma[0])#99th percentile 2.33, 97.5th percentile 1.960
    print('BPnd cut-off used for '+region+' = '+str(upper_cent))       
    region_df = get_subj_status(df, cutoff=upper_cent, region=region, param='BPnd')
    
    if region == 'composite':
        # if it is the composite region then add the subject status to the output dataframe
        df_out['Status'] = region_df.loc[:,'Status']
    
    # calculate z-scores for BPnd based on GMM group 0 mu and sigma (recalc X as previously I removed the NaNs)
    df_region = df.loc[(df['Session']=='baseline') & (df['ROI']==region),['Subject',param+'_srtm', 'R1_srtm']]
    X = df_region[param+'_srtm'].to_numpy()
    X = X[0:(len(X)-1)] #the last datapoint is weird, need to investigate, remove for now
    df_out[region+'_'+param+'_srtm'] = X
    df_out[region+'_BPnd_z']= (X-mu[0])/sigma[0]
    
    #calculate z score for R1 using the patients who are below the BPnd cutoff
    df_ctl = df_region.drop(df_region[region_df.Status == 'PT'].index) # getting only the controls
    # compute the mean and standard deviation of the control population
    Y = df_ctl[['R1_srtm']].to_numpy()
    mean_control = Y.mean()
    std_control = Y.std()
    #here I multiply it all by -1 as blood flow reduces as the disease progresses
    X = df_region['R1_srtm'].to_numpy()
    X = X[0:(len(X)-1)] #the last datapoint is weird, need to investigate, remove for now
    df_out[region+'_R1_srtm'] = X 
    X_z = -1*((X-mean_control)/std_control)
    df_out[region+'_R1_z']= X_z # -1*((X-mean_control)/std_control)
    
    #set zmax by region and by biomarker 
    #for BPnd use the median value of the second peak to define max
    df_zmax.at[df_zmax.loc[df_zmax['Region'] == region].index[0],'BPnd z_max']=(mu[1]-mu[0])/sigma[0]
    
    #for R1 use 3 standard deviations provded that you have at least 10 subjects in the highest score
    if len(X_z[X_z>3])>10:
        R1_max = 5
        R1_z = [1,2,3]
    elif len(X_z[X_z>2])>10:
        R1_max = 3
        R1_z = [1,2]
    else:
        R1_max = 2
        R1_z= [1]

    df_zmax.at[df_zmax.loc[df_zmax['Region'] == region].index[0],'R1 z_max']=R1_max
    df_zmax.at[df_zmax.loc[df_zmax['Region'] == region].index[0],'R1 z']=R1_z   
#write out the complete dataframe with all params    
df_out.to_csv(os.path.join(outpath, 'zscore_allregions_'+ str(components)+'component_'+desc+'_co'+str(cutoff_centile)+'th.csv'))
df_zmax.to_csv(os.path.join(outpath, 'zmax_allregions_'+ str(components)+'component_'+desc+'_co'+str(cutoff_centile)+'th.csv'))
     
    #image_counter = image_counter+1
        #plt.show()