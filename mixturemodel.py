#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 14:18:05 2022

@author: catherinescott
"""
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
param = 'BPnd' #'BPnd'
region_names =['composite','frontal','parietal','insula','temporal','occipital']
components_to_fit = [2] #[1,2,3]
plot_centile = 97.5
cutoff_centile = 97.5

#input/output-----------------------------------------------------------------
#description to add to outputs
desc = '1946-srtm-cleanandAVID27-'+param

#define paths
datapath = '/Users/catherinescott/PycharmProjects/sustain_test/csvfiles'
outpath = '/Users/catherinescott/PycharmProjects/sustain_test/mixturemodel_out'
if not os.path.exists(outpath):
    os.makedirs(outpath)

#load in the csv file comtaining the srtm parameters for each subject
#assuming that you want to use all the csv files in the datapath folder
datacols=['Subject', 'Session','ROI',param+'_srtm']
all_csv_files = glob.glob(os.path.join(datapath, "*.csv"))
df = pd.concat((pd.read_csv(f, skiprows=0,usecols=datacols) for f in all_csv_files), ignore_index=True)

#fitting the GMMs---------------------------------------------------------

#generate multiplication factors to get different centiles
centile_LUT = np.array([[75,0.675],[90,1.282],[95,1.645],[97.5,1.960],[99,2.326]])
plot_factor = centile_LUT[np.where(centile_LUT[:,0]==plot_centile),1].item()
cutoff_factor = centile_LUT[np.where(centile_LUT[:,0]==cutoff_centile),1].item()


image_counter = 0

for region in region_names:

    print('Processing '+ region+'--------------------------------------------')    

    df_out = df.loc[(df['Session']=='baseline') & (df['ROI']==region),['Subject',param+'_srtm']]
    df_out.dropna(axis=0,inplace=True)
    X = df_out[[param+'_srtm']].to_numpy()
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
    
    for j in range(len(components_to_fit)):
    
        # fit Gaussian mixture model
        components = components_to_fit[j]
        
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
        
        #determine whether subjects are controls or not (data is sorted so that group 0 is minimum)
        upper_cent = mu[0]+(cutoff_factor*sigma[0])#99th percentile 2.33, 97.5th percentile 1.960
        print('BPnd cut-off used for '+region+' = '+str(upper_cent))
        region_df = get_subj_status(df, cutoff=upper_cent, region=region, param='BPnd')
        region_df.to_csv(os.path.join(outpath, 'GMM_'+region+'_'+ str(components)+'component_'+desc+'_co'+str(cutoff_centile)+'th.csv'))
        
        image_counter = image_counter+1
        #plt.show()