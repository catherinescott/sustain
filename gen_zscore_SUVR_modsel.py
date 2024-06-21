#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 13:57:13 2024

@author: catherinescott
"""

#this program ruses GMM 1 vs 2 componenets for feature selection (only regions with 2 components will be used)
# and then calculates the z-scores for them

# step 1) read in relevent csvs
# step 2) loop over regions and fit GMM with 1 or 2 components. Use the BIC to determine the better model
# step 3) if the region has 2 components the calculate the z-score for all of the subjects
# step 4) determine the number of z-score levels based on the number of subjects at each level
# step 5) save the resulting csv files


import os
import glob
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# parameters to set/test:
region_names =['composite','frontal','parietal','precuneus','occipital','temporal','insula']
#components_to_fit = 2
plot_centile = 97.5
cutoff_centile = 97.5

ref_regions = ['cereb', 'gm-cereb']
PVC_flags = ['pvc-' ,'']
data_merge_opts = ['followupplus', 'baseline', 'baselineplus', 'all'] 

# step 1) read in relevent csv's----------------------------------------------------------------

image_counter = 0

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
            outpath = out_folder+'/genZscoremodsel_out/'+PVC_flag+ref_region
            if not os.path.exists(outpath):
                os.makedirs(outpath)
                
            
            df = pd.read_csv(os.path.join(datapath, desc+'_sustain_raw.csv'))
            
            
            # step 2) loop over regions and fit GMM with 1 or 2 components. Use the BIC to determine the better model
            
            #generate multiplication factors to get different centiles
            centile_LUT = np.array([[75,0.675],[90,1.282],[95,1.645],[97.5,1.960],[99,2.326]])
            plot_factor = centile_LUT[np.where(centile_LUT[:,0]==plot_centile),1].item()
            cutoff_factor = centile_LUT[np.where(centile_LUT[:,0]==cutoff_centile),1].item()
            
            
            
            
            #define output dataframe for z scores by patient
            #df_out = df.loc[['Subject']]
            df_out = df[['Subject']].copy()
            
            
            #define output dataframe for z_max by region
            df_zmax = pd.DataFrame([region_names,np.ones((len(region_names),1)),np.ones((len(region_names),1)),np.ones((len(region_names),1))*1.960,np.ones((len(region_names),1))]).transpose()
            df_zmax.rename(columns={0: 'Region', 1: 'flow z_max' , 2 : 'amyloid z_max', 3:'amyloid z', 4: 'flow z'}, inplace=True)
            #, columns=['Region','R1 z_max','BP z_max']
            
            for region in region_names:
            
            
                print('Processing '+ region+'************************************')    
            
                params = ['amyloid','flow']
                
                for param in params:
                    
                    print(param+'--------------------------------------------')
                    
                    df_region = df[[region+'_'+param]] 
                    X = df_region.to_numpy()
            
            
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
                        
                        _ = plt.legend(loc='upper right')
                        _ = plt.xlabel(region+' '+param)  
                        _ = plt.ylabel('Probability density')  
                        _ = plt.title('Gaussian mixture model with '+str(components)+
                                          ' components (BIC='+str(round(gmm.bic(X),1))+', AIC='+str(round(gmm.aic(X),1))+')')
                        plt.savefig(os.path.join(outpath,'GMM_'+region+'_'+ param+'_'+str(components)+'component_'+desc+'.pdf'))
                        image_counter = image_counter+1
                        
                        BIC.append(gmm.bic(X))
                        
                        print(str(components)+' comp: BIC='+str(round(gmm.bic(X),1))+', AIC='+str(round(gmm.aic(X),1))+')')
                        
                    # see whether we will use this region or not
                    if BIC[0]<=BIC[1]:
                        # one component model has lower BIC, dont include this region
                        print(region+' '+param+' EXCLUDED as 1 gaussian has lower BIC')
                        df_zmax.at[df_zmax.loc[df_zmax['Region'] == region].index[0],param+' z_max']='NaN'
                        df_zmax.at[df_zmax.loc[df_zmax['Region'] == region].index[0],param+' z']='NaN'
                    else: # two component model has lower BIC
                        print(region+' '+param+' INCLUDED as 2 gaussian has lower BIC')
            # step 3) if the region has 2 components the calculate the z-score for all of the subjects
            
                        # calculate z-scores based on GMM group 0 mu and sigma (recalc X as previously I removed the NaNs)
                        df_region = df[['Subject',region+'_'+param]]
                        #df_region = df.loc[(df['Session']=='baseline') & (df['ROI']==region),['Subject',param+'_srtm', 'R1_srtm']]
                        X = df_region[region+'_'+param].to_numpy()
                    
                        df_out[region+'_'+param+'_suvr'] = X
                        if param == 'flow':
                            X_z = -1*(X-mu[0])/sigma[0]
                        else:
                            X_z = 1*(X-mu[0])/sigma[0]
                        df_out[region+'_'+param+'_z']= X_z #(X-mu[0])/sigma[0]
                        
            
            # step 4) determine the number of z-score levels based on the number of subjects at each level
                        # use 3 standard deviations provded that you have at least 10 subjects in the highest score
                        if len(X_z[X_z>3])>20:
                            R1_max = 5
                            R1_z = [1,2,3]
                        elif len(X_z[X_z>2])>20:
                            R1_max = 3
                            R1_z = [1,2]
                        else:
                            R1_max = 2
                            R1_z= [1]            
            
                        df_zmax.at[df_zmax.loc[df_zmax['Region'] == region].index[0],param+' z_max']=R1_max
                        df_zmax.at[df_zmax.loc[df_zmax['Region'] == region].index[0],param+' z']=R1_z   
            
                    
            
            # step 5) save the resulting csv files
            
            #write out the complete dataframe with all params    
            df_out.to_csv(os.path.join(outpath, 'zscore_allregions_'+desc+'.csv'))
            df_zmax.to_csv(os.path.join(outpath, 'zmax_allregions_'+desc+'.csv'))
             
        













