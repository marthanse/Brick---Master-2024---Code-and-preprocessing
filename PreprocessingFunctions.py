# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 14:28:48 2024

@author: marth
"""
#%%
from spectral import *
import numpy as np
import matplotlib.pyplot as plt
import math
import spectral.io.envi as envi
import spectral as sp
import pandas as pd
import os
from ImportImagesFunctions import *
from SaveLoadDictionariesFunctions import *
from PreprocessingFunctions import *
#%% Preprocessing functions

def CutTop(Image):
    """
    Parameters
    ----------
    Image : Single image which you want to cut,

    Returns
    -------
    CutImage : The image after cutting of the top.

    """
    # row number sat to 300, since this is suitable for all images.
    CutImage = Image[300:,:,:]
    return CutImage


def CutTopofImages(BilFileDict):
    '''
    Cut unecessary area on top of the images, above the white reference.
    
    '''
    CutDict = {}
    for key in BilFileDict:
        CutDict[key] = CutTop(BilFileDict[key])
    return CutDict


def WhiteCorrection(image, end_white_ref):
    """
    Correct for white reference with reflectance at 60%
    Parameters
    ----------
    image: original image, with white ref included in image. 
    end_white_ref: integer, the last row of the white ref in the image

    Returns
    -------
    corr_img : white corrected images
    """
    rows = image.shape[0]
    cols = image.shape[1]
    bands=image.shape[2]

    means = np.empty([cols, bands])

    for i in range(cols):
        for band in range(bands):
            # legger til nanmean istedenfor nan
            means[i, band] = np.nanmean(image[:end_white_ref, i, band]) 

    corr_img = np.empty([rows, cols, bands])
    for i in range(cols):
        for band in range(bands):
            corr_img[:, i, band] = (image[:, i, band] / means[i][band]) * 0.60
    
    return corr_img


def WhiteCorrectAllImages(BilFileDict):
    """
    NB: end white ref is currently 400 for all images

    Parameters
    ----------
    BilFileDict : Dictionary with all uncorrected images

    Returns
    -------
    CorrDict : Dictionary with all corrected images

    """
    CorrDict = {}
    
    for key in BilFileDict:
        CorrDict[key] = WhiteCorrection(BilFileDict[key], 400)
    # Filnavn og mappe må byttes for untreated og dried!
    SaveDictNumpy(r'D:\Whitecorrected images\untreated', CorrDict, '_whitecorrected_image')
    return CorrDict

def FindBrickStart(image, rows):
    """

    Parameters
    ----------
    image : the image to be cut
    rows : number of rows in the image

    Returns
    -------
    BrickStart : the pixel row at col 0 and band 0 where the brick begins

    """
    #rows = image.shape[0] 
    for j in range(rows):
            # if img[j,0,0] < 0.1:
                # added a condition to avoid nan trouble
                if image[j, 0, 0] < 0.3 and not np.isnan(image[j, 0, 0]):
                    BrickStart = j + 20
                    break

    return BrickStart

def CutWhiteRef(image):
    """
    Parameters
    ----------
    image : image which has a white reference to be removed

    Returns
    -------
    CutImage : image with the white reference part removed

    """
    rows = image.shape[0] 
    
    BrickStart = FindBrickStart(image, rows)
    CutImage = image[BrickStart:, :, :]
    
    return CutImage

    
def CutAllWhiteRef(CorrectedDict):
    """
    Parameters
    ----------
    CorrectedDict : Dictionary containing all images which has a white ref to remove.
    Returns
    -------
    CuttedDict : Dictionary with all images without white ref, after removal

    """
    CuttedDict = {}
    
    for key in CorrectedDict:
        CuttedDict[key] = CutWhiteRef(CorrectedDict[key])

    return CuttedDict

def BackgroundFilter(image):
    th = 0.06  # denne verdien fjerner nesten all bakgrunn nederst og langs siden i tillegg til tusj.
    x, y, z = image.shape

    background_filter = []
    for j in range(x):
        filter_rows = []
        for i in range(y):
            if image[j, i, 130] < th:   # må bestemme en bestemt bølgelengde treshholden skal hentes fra! obs!
                filter_rows.append(np.nan) #endrer verdiene fra False til NaN i håp om at det funker
            elif np.isnan(image[j, i, 130]): # håper dette stemmer
                filter_rows.append(np.nan)
            else:
                filter_rows.append(True)
        background_filter.append(filter_rows)
    background_filter = np.array(background_filter)  # hvorfor gjorde emilie dette?
    
  #  print(background_filter.shape)
    #plt.figure(figsize=(5, 5))
    #plt.imshow(background_filter) #Print dark filter to check it looks fine
    #plt.show()
    
    return background_filter

def BackgroundFilterMultipleBricks(ImageDict):
    
    BackgroundFilterDict = {}
    
    for key in ImageDict:
        BackgroundFilterDict[key] = BackgroundFilter(ImageDict[key])
    return BackgroundFilterDict


def FindBottomBrick(background_filter):
    x, y = background_filter.shape
   #print(x,y)
    BottomBrick = 0
                
    for i in range(x):
        check = np.all(np.isnan(background_filter[i, :]))
       # print(check)
        if check == True and i > 400:
           # print('check initiated')
            #BottomBrick = i
            BottomBrick = i - 50
            break
        else:
            #BottomBrick = i
            BottomBrick = i - 50
    return BottomBrick
                

def CutBottomBrick(image, background_filter): 
    bottom_brick = FindBottomBrick(background_filter)
    CutImage = image[0:bottom_brick, :, :]    
    return CutImage


def CutBottomBackgroundFilter(background_filter):
    bottom_brick = FindBottomBrick(background_filter)
    cut_filter = background_filter[0:bottom_brick, :]
   # print(cut_filter.shape)
    #plt.figure(figsize=(5, 5))
    #plt.imshow(cut_filter) #Print dark filter to check it looks fine
    #plt.show()
    
    return cut_filter


def CutAllBottoms(image_dict, backgroundfilter_dict):
    cutted_dict = {}
    for key in image_dict:
        cutted_dict[key] = CutBottomBrick(image_dict[key], backgroundfilter_dict[key])
    
    return cutted_dict  

def CutAllFilters(filter_dict):
    cutted_dict = {}
    for key in filter_dict:
        cutted_dict[key] = CutBottomBackgroundFilter(filter_dict[key])
    
    return cutted_dict   

    
        

if __name__ == "__main__":
    
    
    
    ## Tester FindBrickBottom
    folder1 = r"D:\test_A_brick"
    ImageHeaders = CollectOnlyHeaders(folder1)
    header = ImageHeaders['A9_A']
    ww = Wavelengths(ImageHeaders, folder1)
    sat_images = SaturationFilterAllImages(folder1, ImageHeaders)
    sat_image = FindOversaturation(folder1, header)
    cut_imagetops = CutTopofImages(sat_images)
    cut_image = CutTop(sat_image)

    ShowImage(cut_image)    
    # WhiteCorrection
    corrected_images = WhiteCorrectAllImages(cut_imagetops)
    corr_image = WhiteCorrection(cut_image, 400)
    corr_image = CutWhiteRef(corr_image)
    ShowImage(corr_image)
    images = CutAllWhiteRef(corrected_images)
    background_filters =  BackgroundFilterMultipleBricks(images)
    background_filter = BackgroundFilter(corr_image)
    
    plt.figure()
    plt.imshow(final_filter)
    plt.show()
    
    only_bricks = CutAllBottoms(images, background_filters)
    final_image = CutBottomBrick(corr_image, background_filter)
    final_filter = CutBottomBackgroundFilter(background_filter)

    
    
