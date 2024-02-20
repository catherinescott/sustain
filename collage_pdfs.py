#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 17:20:55 2022

@author: catherinescott
"""
from PIL import Image
from PIL import ImageDraw, ImageFont
from pdf2image import convert_from_path
im_width=1280
im_height=960
col_width = im_width*3
col_height = im_height*2
collage = Image.new("RGBA", ((col_width),(col_height)))


subject_names = ['AVID2001','AVID2002','AVID2003','AVID2005','AVID2006','AVID2007','AVID2008','AVID2009', 'AVID2010','AVID2011','AVID2012','AVID2013','AVID2014','AVID2015','AVID2016','AVID2017']
w_names = ['dur', 'sqrt_dur','dur_decay','dur_decay_neg','decay']

title_font = ImageFont.truetype('/Library/Fonts/Microsoft/Calibri.ttf', 100)
    
    
#while k<(len(w_names)-1):
for subj in subject_names:
    k=0
    
    for i in range(0,col_height,im_height):
        for j in range(0,col_width,im_width):
            new_img = convert_from_path("/Users/catherinescott/Documents/cluster/AVID2/analysis/srtm-pct-gif-cerebgm-basis-avid2-weighttest/sub-"+subj+"/ses-baseline/pet/sub-"+subj+"_ses-baseline_desc-pet_"+w_names[k]+"-composite-srtm-plot.pdf")
            new_img = new_img[0]
            edit_img = ImageDraw.Draw(new_img)
            edit_img.text((15,15), w_names[k], (0, 0, 0),font=title_font)
            
            collage.paste(new_img, (j,i))
        
            print(k)
            k+=1
            if k>(len(w_names)-1):
                break
        if k>(len(w_names)-1):
            break
    collage.show()
    collage.save("/Users/catherinescott/Documents/SuStaIn_out/collage_pdfs/"+subj+"_weights.png")
    #collage.close()