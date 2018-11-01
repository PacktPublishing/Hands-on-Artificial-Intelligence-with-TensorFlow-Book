'''
Created on 29-Sep-2018

@author: DX
'''

import cv2
import glob

cords = [320,240]
h,w = 380,650

# cords = [520,570]
# h,w = 490,740

def crop(im):    
    cropped = im[cords[0]:(cords[0]+h),cords[1]:(cords[1]+w)]
    cropped = cv2.resize(cropped,(96,64))
    return cropped