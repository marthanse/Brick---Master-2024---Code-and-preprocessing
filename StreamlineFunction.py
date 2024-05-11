# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:41:08 2024

Function to streamline image preprocessing

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
#import skimage as ski
from MakeBlocksFunctions import *
from StreamlineFunction import *
from SaveLoadDictionariesFunctions import *
from SNVProcessingFunctions import *
from MSCProcessingFunctions import *
from scipy.signal import savgol_filter

#%%
def StreamlineBrickOutline(folder):
    """
    Parameters
    ----------
    folder : the folder path to images being analyzed, from the original image files

    Returns
    -------
    only_bricks : the outlines of all the bricks
    background_filters_corrected : all the background filters with true or false values

    """
    # Retrieve the strings for headerfiles
    header_files = CollectOnlyHeaders(folder)
    
    # Give the image values with oversaturation value NaN
    ##sat_images = SaturationFilterAllImages(folder, image_headers)
    images = SaturationFilterAllImages(folder, header_files)
    
    # Collect the wavelengths
    ww = Wavelengths(header_files, folder)
    
    # Cut of unecessary background of all images above the white plate
    #cut_imagetops = CutTopofImages(image_files)
    #cut_imagetops = CutTopofImages(sat_images)
    images = CutTopofImages(images)
    
    # WhiteCorrection
    # corrected_images = WhiteCorrectAllImages(cut_imagetops)
    images = WhiteCorrectAllImages(images)
    
    #Cut of the white reference
   # images_without_whiteref=CutAllWhiteRef(corrected_images)
    images = CutAllWhiteRef(images)
    
    # Retrieve the background filters
    #background_filters =  BackgroundFilterMultipleBricks(images_without_whiteref)
    background_filters =  BackgroundFilterMultipleBricks(images)
    
    # Be left with only brick outlines and cut the backgroundfilters to fit
    #only_bricks = CutAllBottoms(images_without_whiteref, background_filters)
    only_bricks = CutAllBottoms(images, background_filters)
    
    #background_filters_corrected = CutAllFilters(background_filters)
    background_filters = CutAllFilters(background_filters)
    
    #return only_bricks, background_filters_corrected, ww
    return header_files, only_bricks, background_filters, ww


def StreamlinewithWhiteCorrImage(header_folder, whitecorrected_image_folder):
    """
    Parameters
    ----------
    folder : the folder path to images being analyzed, from already whitecorrected images

    Returns
    -------
    only_bricks : the outlines of all the bricks
    background_filters_corrected : all the background filters with true or false values

    """
    
    # Retrieve the strings for headerfiles
    header_files = CollectOnlyHeaders(header_folder)
    
    # Open the whitecorrected imagefiles
    images = OpenDictNumpy(whitecorrected_image_folder, header_files, '_whitecorrected_image')
    
    # Collect the wavelengths
    ww = Wavelengths(header_files, header_folder)
    
    #Cut of the white reference
    images = CutAllWhiteRef(images)
    
    # Retrieve the background filters
    background_filters =  BackgroundFilterMultipleBricks(images)
    
    # Be left with only brick outlines and cut the backgroundfilters to fit
    only_bricks = CutAllBottoms(images, background_filters)
    background_filters = CutAllFilters(background_filters)
    
    return header_files, only_bricks, background_filters, ww

def StreamlineMortarRemoved(header_folder, mortar_image_folder):
    """
    Streamline from white corrected images with removed white references,
    that have gotten mortar covered areas removed by the MortarRemovalFunctions.
    The images with mortar removed has been presaved.
    """
    
    # Retrieve the strings for headerfiles
    header_files = CollectOnlyHeaders(header_folder)
    
    # Open the whitecorrected imagefiles
    images = OpenDictNumpy(mortar_image_folder, header_files, '_nomortar')
    
    # Collect the wavelengths
    ww = Wavelengths(header_files, header_folder)
    
    # Retrieve the background filters
    background_filters =  BackgroundFilterMultipleBricks(images)
    
    # Be left with only brick outlines and cut the backgroundfilters to fit
    only_bricks = CutAllBottoms(images, background_filters)
    background_filters = CutAllFilters(background_filters)
    
    return header_files, only_bricks, background_filters, ww

def StreamlineMeanSpectra(brick_images, wavelengths, background_filters):
    """
    Section the brick into 12 sections of equal height and width. 
    Generates mean spectra for each section. Each brick side ends up with 12 representative mean spectra.
    
    Parameters
    ----------
    brick_images : the processed brick images with background removed
    wavelengths : list of wavelength bands
    background_filters : the pixels to include or not based on background or not background

    Returns
    -------
    mean_spectra_all_bricks_w_blocks : for each brick image, 12 mean spectra are generated and added to a dictionary

    """
    mean_spectra_all_bricks_w_blocks = BlockAllBricks(brick_images, wavelengths, background_filters)
    return mean_spectra_all_bricks_w_blocks

def PlotRawSpectra(mean_spectra_dict, ww):
    
    for key in mean_spectra_dict:
        counter = 1
        plt.figure(figsize=(10, 10))
        for spectra in range(len(mean_spectra_dict[key])):
            plt.plot(ww, mean_spectra_dict[key][spectra], label=f'{counter}')
            counter+= 1
        plt.title(f"Mean reflectance spectra for each of the 12 areas of brick {key}", fontsize=20)    
        plt.legend(title='Area', title_fontsize=17, fontsize = 13)
        plt.xlabel('Wavelength [nm]', fontsize = 17)
        plt.ylabel('Reflectance', fontsize = 17)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
       # plt.savefig(f'{key}_mean_raw_spectra_dried_no_mortar.png') # Save figure with 12 spectra per brick

def PlotSGSpectra(mean_spectra_dict, ww):
    
    for key in mean_spectra_dict:
        counter = 1
        plt.figure(figsize=(10, 10))
        for spectra in range(len(mean_spectra_dict[key])):
            plt.plot(ww, mean_spectra_dict[key][spectra], label=f'{counter}')
            counter+= 1
        plt.title(f"Mean reflectance spectra for each of the 12 areas of brick {key} after Savitzky-Golay filtering", fontsize=20)    
        plt.legend(title='Area', title_fontsize=17, fontsize = 13)
        plt.xlabel('Wavelength [nm]', fontsize = 17)
        plt.ylabel('Reflectance', fontsize = 17)  
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
      #  plt.savefig(f'{key}_mean_spectra_SG_1103.pdf')
        
def PlotAbsSpectra(mean_spectra_dict, ww):
    
    for key in mean_spectra_dict:
        counter = 1
        plt.figure(figsize=(10, 10))
        for spectra in range(len(mean_spectra_dict[key])):
            plt.plot(ww, mean_spectra_dict[key][spectra], label=f'{counter}')
            counter+= 1
        plt.title(f"Mean absorption spectra for each of the 12 areas of brick {key}", fontsize=20)    
        plt.legend(title='Area', title_fontsize=17, fontsize = 13)
        plt.xlabel('Wavelength [nm]', fontsize = 17)
        plt.ylabel('Absorption', fontsize = 17)  
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
      #  plt.savefig(f'{key}_mean_abs_spectra_1103.pdf')
        
def PlotSNVSpectra(mean_spectra_dict, ww):
    
    for key in mean_spectra_dict:
        counter = 1
        plt.figure(figsize=(15, 10))
        for spectra in range(len(mean_spectra_dict[key])):
            plt.plot(ww, mean_spectra_dict[key][spectra], label=f'{counter}')
            counter+= 1
       # plt.title(f"Mean absorption spectra for each of the 12 areas of brick {key} after SNV", fontsize=20)    
       # plt.legend(title='Area', title_fontsize=30, fontsize = 28)
        plt.xlabel('Wavelength [nm]', fontsize = 35, fontweight='bold')
        plt.ylabel('Absorption', fontsize = 35, fontweight='bold')
        plt.xticks(fontsize=35, fontweight='bold')
        plt.yticks(fontsize=35, fontweight='bold')
        plt.tick_params(axis='both', which='major', length=10, width=3)
        #plt.savefig(f'{key}_dried_SNV_spectra_mortar_removed.png')
        
if __name__ == "__main__":
    
    
    
    
    #%% Test mortar removed images, 07.04
    header_folder = r'D:\torre\Gruppe 8 SVB'
    mortar_image_folder = r"D:\mortardetected images\untreated"
    whitecorr_folder = r"D:\Whitecorrected images\dried"
    header_files = CollectOnlyHeaders(header_folder)
    images = OpenDictNumpy(whitecorr_folder, header_files, '_whitecorrected_image')
    images = CutAllWhiteRef(images)
    ShowImages(images)
    ww = Roundww(Wavelengths(header_files, header_folder))
    header_files, bricks, filters, ww = StreamlineMortarRemoved(header_folder, mortar_image_folder)
    header_files, bricks, filters, ww = StreamlinewithWhiteCorrImage(header_folder, whitecorr_folder)
    
    for filter in filters:
        plt.figure(figsize=(5, 5))
        plt.imshow(filters[filter]) #Print dark filter to check it looks fine
        plt.show()  
        
    ShowImages(bricks)
    raw_spectra = StreamlineMeanSpectra(bricks, ww, filters)
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v3 edge and mortar removal\spectra after mortar removal\dried raw spectra', 
                  raw_spectra, '_spectra')
  
    PlotRawSpectra(raw_spectra, ww)

    
    #%% Band selection
    header_folder = r'D:\untreated\Gruppe 8 SVB'
    whitecorr_folder = r"D:\Whitecorrected images\untreated"
    header_files = CollectOnlyHeaders(header_folder)
    
    # Bands
    selected_mortar_bands = [257, 249, 250, 251, 252, 253, 254, 255, 256, 181, 182, 183, 184, 185, 186, 84]
    few_mortar_bands = [256, 186, 84]
    selected_brick_bands = [174, 175, 229, 230, 231, 232, 233, 234]
    testing = [0, 49, 119, 287]
    
    ww_nm = Roundww(Wavelengths(header_files, header_folder))
    images = OpenDictNumpy(whitecorr_folder, header_files, '_whitecorrected_image')
    images=CutAllWhiteRef(images)
    image = images['SVB1_B']

    for i, band_index in enumerate(testing):
        selected_image = image[:, :, band_index]
        plt.figure()
        plt.imshow(selected_image, cmap='gray') 
        plt.title(f'Intensity {ww_nm[band_index]} nm')
        plt.colorbar()
        plt.axis('off')

        plt.tight_layout()
        plt.show()
        
    for i, band_index in enumerate(few_mortar_bands):
        selected_image = image[:, :, band_index]
        plt.figure()
        plt.imshow(selected_image, cmap='gray', clim=(0,0.8))  
        #plt.title(f'Reflectance {ww_nm[band_index]} nm')
        plt.axis('off')
        plt.colorbar().ax.tick_params(labelsize=30)
        plt.tight_layout()
        plt.show()
        # the gray cmap works so you should look for the brighest areas 
    
 
