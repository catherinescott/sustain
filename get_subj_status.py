#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 11:16:52 2022

@author: catherinescott
"""

#this function reads in the dataframe containing the binding potential values
#and outputs a dataframe with just the subject ID and status
def get_subj_status(df_in, cutoff=0.35, region='composite', param='BPnd'):

    import numpy as np 
    
    df_out = df_in.loc[(df_in['Session']=='baseline') & (df_in['ROI']==region),['Subject',param+'_srtm']]
    #label all patients 'PT' if BPnd of the composite region is greater than cutoff or 'CTL' otherwise
    df_out['Status'] = np.where(df_out['BPnd_srtm']<=cutoff,'CTL','PT')
    df_out.drop(param+'_srtm', axis=1, inplace=True)
    
    return df_out