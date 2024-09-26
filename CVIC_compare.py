#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:05:36 2024

@author: catherinescott
"""

import os
from PIL import Image
from PIL import ImageDraw, ImageFont
from pdf2image import convert_from_path




#title_font = ImageFont.truetype('/Library/Fonts/Microsoft/Calibri.ttf', 100)

ref_regions = ['gm-cereb','cereb'] #['cereb']#
PVC_flags = ['pvc-','']#['pvc-' ,'']
data_merge_opts = ['baseline', 'baselineplus','followupplus', 'all'] #['followupplus', 'baseline', 'baselineplus', 'all'] 
include_biomarker_list = [['amyloid','flow'],['flow'],['amyloid']]#[['flow'],['amyloid'],['flow','amyloid']]

remove_zero_subs='no'

im_width=1280
im_height=960
col_width = im_width*len(PVC_flags)
col_height = im_height*len(data_merge_opts)

collage = Image.new("RGBA", ((col_width),(col_height)))
fontsize = 30
font = ImageFont.truetype("/System/Library/Fonts/Supplemental/arial.ttf", fontsize)

for ref_region in ref_regions:
    for include_biomarkers in include_biomarker_list:
        
        data_merge_idx = 0
         
        for i in range(0,col_height,im_height):
            
            PVC_idx = 0
            for j in range(0,col_width,im_width):
                
                if PVC_idx>len(PVC_flags)-1:
                    break
                if data_merge_idx>len(data_merge_opts)-1:
                    break               
                
                data_merge_opt=data_merge_opts[data_merge_idx]
                PVC_flag = PVC_flags[PVC_idx ]                
            
                print('Running: ref: '+ref_region+', PVC: '+PVC_flag+', datamerge: '+data_merge_opt+' biomarkers:'+' '.join(include_biomarkers))
                
                cmmt = PVC_flag+ref_region+'_'+ data_merge_opt
                test_run = 'run'
                ## data in--------------------------------------------------------------------
                #descriptions to use for input and output data
                
                in_desc = '1946AVID2YOADSUVR_v1'#'1946-srtm-cleanandAVID27'
                if remove_zero_subs=='yes':
                    out_desc = in_desc+'-GMM_'+'_'.join(include_biomarkers)+'_'+test_run+'_'+cmmt+'removezero_v1'
                else:
                    out_desc = in_desc+'-GMM_'+'_'.join(include_biomarkers)+'_'+test_run+'_'+cmmt+'_v1'
                
                
                out_folder = '/Users/catherinescott/Documents/python_IO_files/SuStaIn_test/SuStaIn_out'
                outpath = out_folder+'/run_SuStaIn_GMM/'+PVC_flag+ref_region
                
                dataset_name = out_desc
                output_folder = outpath+'/'+dataset_name
                CVIC_plot_path = os.path.join(output_folder,'CVIC_'+out_desc+'.pdf')
                if os.path.isfile(CVIC_plot_path):
                    new_img = convert_from_path(CVIC_plot_path)
                    new_img = new_img[0]
                    edit_img = ImageDraw.Draw(new_img)
                    edit_img.text((15,15), 'ref: '+ref_region+', PVC: '+PVC_flag+', datamerge: '+data_merge_opt+', biomarkers:'+' '.join(include_biomarkers), (0, 0, 0),font=font)
                    
                    collage.paste(new_img, (j,i))
                else:
                    print('file missing: '+CVIC_plot_path)
                print('data_merge_idx '+str(data_merge_idx)+' PVC_idx: '+str(PVC_idx))
                
                PVC_idx = PVC_idx+1
            data_merge_idx=data_merge_idx+1
                
        collage.show()
        collage.save(out_folder+'/run_SuStaIn_GMM/'+ref_region+'_'+'_'.join(include_biomarkers)+'.png')
        