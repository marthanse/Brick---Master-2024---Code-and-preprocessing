# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 10:48:52 2024

@author: marth
"""
#%% Import packages
from spectral import *
import numpy as np
import matplotlib.pyplot as plt
import math
import spectral.io.envi as envi
import spectral as sp
import pandas as pd
import os

#%%

def CollectImages(folder_path):
    """
    Creates dictonary used for image import
    Parameters
    ----------
    folder_path : the name of the relevant folder with the images

    Returns
    -------
    images : a dictinary conatining keys with distinct image id's and the file name'

    """
    images = {}
    headers = {}
    
    for file in os.listdir(folder_path):
        if file.endswith(".img"):
            # Extract the four first letters of the file name to identify brick
            var_name = file.split("_SWIR")[0]
            # Store filename in the dictionary with the variable name as key
            images[var_name] = file
        if file.endswith("float32.hdr"):
            var_name = file.split("_SWIR")[0]
            headers[var_name] = file
    return images, headers

def CollectOnlyHeaders(folder_path):
    """
    Creates dictonary used for image import
    Parameters
    ----------
    folder_path : the name of the relevant folder with the images

    Returns
    -------
    images : a dictinary conatining keys with distinct image id's and the file name'

    """
    headers = {}
    
    for file in os.listdir(folder_path):
        if file.endswith("float32.hdr"):
            var_name = file.split("_SWIR")[0]
            headers[var_name] = file
    return headers
            
def ImportImages(folder, image_dict, header_dict):
    
    BilFileDict = {}
    
    for key in image_dict:
        
        img = envi.open(folder+'\\'+header_dict[key], folder+'\\'+image_dict[key])
        BilFileDict[key] = img
    return BilFileDict

def Wavelengths(HeadersDict, folder_path):    
    """
    Retrieve the wavelengths bands
    """
    counter = 0
    if counter == 0:
        for key in HeadersDict:
            wavelength = envi.read_envi_header(folder_path+'\\'+ HeadersDict[key])['wavelength']
            ww = [float(i) for i in wavelength]
            counter += 1
    return ww

def ShowImage(image):
    imshow(image, (5, 10, 25), stretch=((0.02, 0.98), (0.02, 0.98), (0.02, 0.98)))
    
def ShowImages(ImageDict):
    for key in ImageDict:
        imshow(ImageDict[key], (5, 10, 25), stretch=((0.02, 0.98), (0.02, 0.98), (0.02, 0.98)))

def FindDataIgnoreValue(folder, header_file):
    """
    Parameters
    ----------
    folder : folder_path
    header_file : a .hdr file

    Returns
    -------
    data_ignore_value : the value for which to replace with nan value, non valid value

    """
    envi_file = envi.open(folder+'\\'+header_file)
    data_ignore_value = float(envi_file.metadata["data ignore value"])
    
    return data_ignore_value

def FindOversaturation(folder, header_file):
    """

    Parameters
    ----------
    folder : folder path
    header_file : .hdr file for the relevant image 

    Returns
    -------
    new_array : array with the values of the image, but 
    with oversaturated pixels set to NaN

    """
    data_ignore_val = FindDataIgnoreValue(folder, header_file)
    envi_file = envi.open(folder+'\\'+header_file)
    memmap_data = envi_file.open_memmap()
    where_oversat = np.where(memmap_data == data_ignore_val, 1,0)
    where_oversat_sum = where_oversat.sum(axis=2)
    # where there is oversaturation, it is set to true, else false
    where_oversat_2d = np.where(where_oversat_sum > 0, True, False)
    
    new_array = memmap_data.copy()
    new_array[where_oversat_2d,:] = np.nan
    return new_array


def SaturationFilterAllImages(folder, header_dict):
    """

    Parameters
    ----------
    folder : folder path for headers
    header_dict : dict with the .hdr file names

    Returns
    -------
    saturation_dict : dict with imagenames as keys, where the oversaturated values are nan values

    """
    saturation_dict = {}
    for key in header_dict:
        saturation_dict[key] = FindOversaturation(folder,header_dict[key])
        #print(np.count_nonzero(np.isnan(saturation_dict[key])))
    return saturation_dict

if __name__ == "__main__":
    
    folder = r"D:\test_A_brick"
    ImageStrings, ImageHeaders = CollectImages(folder)
    ImageHeaders = CollectOnlyHeaders(folder)
    ActualImages = ImportImages(folder, ImageStrings, ImageHeaders)
    ShowImages(ActualImages)
    ww = Wavelengths(ImageHeaders, folder)
    ign_val = FindDataIgnoreValue(folder, ImageHeaders['A4_A'])
    image_file_w_nan = FindOversaturation(folder, ImageHeaders['A9_A'])
    # ---> for ut verdier med 0 og 1
    sat_images = SaturationFilterAllImages(folder, ImageHeaders)
    #print(np.count_nonzero(np.isnan(image_file_w_nan)))
    ShowImage(sat_images['A4_A'])

