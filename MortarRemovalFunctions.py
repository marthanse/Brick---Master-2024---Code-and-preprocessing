# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 13:40:21 2024

Mortar detection and removal functions

@author: marth
"""
from spectral import *
import numpy as np
import matplotlib.pyplot as plt
import math
import spectral.io.envi as envi
import spectral as sp
import pandas as pd
import os
from ImportImagesFunctions import *
from PreprocessingFunctions import *
from MakeBlocksFunctions import *
from StreamlineFunction import *
from SaveLoadDictionariesFunctions import *
from SNVProcessingFunctions import *
from MSCProcessingFunctions import *
from scipy.signal import savgol_filter
from sklearn.pipeline import make_pipeline
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from MortarRemovalFunctions import *

#%% Functions

def LoadMortarTrainingData():
    """
    Load the manually made training data for spectra representing mortar and no mortar.
    Made from group 2, 3, 4, 5, and 9.
    """
    mortar_4_data = pd.read_csv("mortar 4")
    nomortar_4_data = pd.read_csv("no mortar 4")
    mortar_5_data = pd.read_csv("mortar 5")
    nomortar_5_data = pd.read_csv("no mortar 5")
    mortar_9_data = pd.read_csv("mortar 9")
    nomortar_9_data = pd.read_csv("no mortar 9")
    mortar_7_data = pd.read_csv("mortar 7")
    nomortar_7_data = pd.read_csv("no mortar 7")
    mortar_2_data = pd.read_csv("mortar 2")
    nomortar_2_data = pd.read_csv("no mortar 2")
    nomortar_3_data = pd.read_csv("no mortar 3")

    df = pd.concat([mortar_4_data, nomortar_4_data, mortar_5_data, nomortar_5_data,
                    mortar_9_data, nomortar_9_data, mortar_7_data, nomortar_7_data, 
                    mortar_2_data, nomortar_2_data, nomortar_3_data], ignore_index=True)
    return df

def TrainPLSmodel():
    """
    Train PLS-DA model for mortar detection

    Parameters
    ----------
    df : dataframe with training data

    Returns
    -------
    pls model

    """
    df = LoadMortarTrainingData()
    
    X_train = df.iloc[:, 0:288]
    y_train = df.iloc[:, 288:290]
    
    # Get the wavelengths
    header_folder = r'D:\untreated\Gruppe 5 BB Standard'
    headers = CollectOnlyHeaders(header_folder)
    ww = Roundww(Wavelengths(headers, header_folder))  # wavelengths same for all images
    
    # Standardize
    sc = StandardScaler()
    df_sc = pd.DataFrame(sc.fit_transform(X_train),columns = ww)
    
    # Train model
    pls = PLSRegression(n_components=5)
    pls.fit(X_train, y_train)
    
    return pls
    

def MortarDetectionFilter(image, pls):
    """
    Detect mortar pixel by pixel in a brick image

    Parameters
    ----------
    image : brick image, with removed whiteref
    pls : trained pls model, fitted to training data

    Returns
    -------
    final_labels : the filter with 1 for mortar and 0 for no mortar

    """
    rows = image.shape[0]
    cols = image.shape[1]
    bands = image.shape[2]
    labels = np.empty([rows, cols])
    for i in range(rows):
        for j in range(cols):
            if not np.isnan(image[i,j]).any():
                values = pls.predict(image[i, j].reshape(1,-1))
                if values[0, 0]>values[0, 1]:  # values[0] is no mortar, values[1] is mortar
                    labels[i, j] = 0  # it is not mortar
                else:
                    labels[i, j] = 1  # it is mortar
            else: 
                labels[i, j] = 1  # if the value is Nan, remove it anyway
    final_labels = labels
    
    return final_labels

# def MortarDetectedImage(image, pls):
#     """
#     Detect mortar pixel by pixel in a brick image

#     Not used in further processing.
#     Parameters
#     ----------
#     image : brick image, with removed whiteref
#     pls : trained pls model, fitted to training data

#     Returns
#     -------
#     final_labels : the filter with 1 for mortar and 0 for no mortar

#     """
#     rows = image.shape[0]
#     cols = image.shape[1]
#     bands = image.shape[2]
#     result = np.copy(image)
#     #detected_image = np.empty([rows, cols, bands])
#     for i in range(rows):
#         for j in range(cols):
#             if not np.isnan(image[i,j]).any():
#                 values = pls.predict(image[i, j].reshape(1,-1))
#                 if values[0, 0]<values[0, 1]:  # values[0] is no mortar, values[1] is mortar
#                     result[i, j, :] = np.nan # it is not mortar
    
#     return result

# def MortarDetectDict(image_dict, pls_model):
#     detected = {}
#     for key in image_dict:
#         new_image = MortarDetectedImage(image_dict[key], pls_model)
#         detected[key] = new_image
#     return detected

def apply_filter_to_image(image, labeled_image):
    """
    Apply a filter to an image and turn matching pixels into NaN values across all bands.

    Parameters
    ----------
    image : numpy array
        Input image.
    filter : numpy array
        Filter with 1's indicating positions to turn into NaN values.

    Returns
    -------
    result : numpy array
        Image with matching pixels replaced by NaN across all bands.
    """
    result = np.copy(image)  # Create a copy of the input image to avoid modifying the original

    # Iterate over the image and the filter simultaneously
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if labeled_image[i, j] == 1:  # Check if the filter value is 1
                result[i, j, :] = np.nan  # Replace all bands of the corresponding pixel in the image with NaN

    return result
        

if __name__ == "__main__":
    

    pls = TrainPLSmodel()
    folder = r'D:\untreated\Gruppe 9 Fornebu'
    headers= CollectOnlyHeaders(folder)
    images = OpenDictNumpy(r'D:\Whitecorrected images\untreated', headers, '_whitecorrected_image')
    images = CutAllWhiteRef(images)
    
   
    fv1a = images['FV1_A']
    #for image in images:
   # new_image = MortarDetectedImage(image, pls)
   # new_image = MortarDetectedImage(fv1a, pls)
    labeled = MortarDetectionFilter(fv1a, pls)
    ShowImage(new_image)  
    
    plt.figure()
    plt.imshow(labeled)
    plt.show()
    
    #%% Check if it is applicable for backgroundfilter and so on
    #new_fornebu = MortarDetectDict(images, pls)
    #ShowImages(new_fornebu)
    
    #%% Try to apply based on the labeled data
    test_brick = images['FV6_B']
    labeled = MortarDetectionFilter(test_brick, pls)
    detected_brick = apply_filter_to_image(test_brick, labeled)
    
    plt.figure()
    plt.imshow(labeled)
    plt.show()
    
    ShowImage(test_brick)
    #%% Apply mortar detection to all bricks
    # Images needs to be changed for each group
    
    for key in images:
        labeled = MortarDetectionFilter(images[key], pls)
        detected_image = apply_filter_to_image(images[key], labeled)
        np.save(r'D:\mortardetected images\untreated'+'\\'+key+'_nomortar'+'.npy', detected_image)
    
 
