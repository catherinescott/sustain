#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 11:16:52 2022

@author: catherinescott
"""

#this function reads in the dataframe containing the binding potential values
#and outputs a dataframe with just the subject ID and status
def get_subj_status_SUVR(df_in, cutoff=0.35, region='composite', param='amyloid'):

    import numpy as np 
    
    #df_out = df_in.loc[(df_in['Session']=='baseline') & (df_in['ROI']==region),['Subject',param+'_srtm']]
    df_out = df_in[['Subject',region+'_'+param]]
    
    #label all patients 'PT' if BPnd of the composite region is greater than cutoff or 'CTL' otherwise
    df_out['Status'] = np.where(df_out[region+'_'+param]<=cutoff,'CTL','PT')
    df_out.drop(region+'_'+param, axis=1, inplace=True)
    
    return df_out