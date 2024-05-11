# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:56:44 2024

@author: marth
"""

import pandas as pd
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
from scipy.signal import savgol_filter
from SNVProcessingFunctions import *
from MSCProcessingFunctions import *

#%%

# Code for MSC found at: https://towardsdatascience.com/scatter-correction-and-outlier-detection-in-nir-spectroscopy-7ec924af668
# Created by Shravankumar Hiregoudar, adapted for this thesis
# Code inspired by Emilie G. Langeland at: https://github.com/moaemilie/Master_preprosessering/blob/main/preprosessering_MSC.py


def msc(input_data, reference=None):
    """
        :msc: Scatter Correction technique performed with mean of the sample data as the reference.        
        :param input_data: Array of spectral data
        :type input_data: DataFrame        
        :returns: data_msc (ndarray): Scatter corrected spectra data
    """
    eps = np.finfo(np.float32).eps
    input_data = np.array(input_data, dtype=np.float64)
    ref = []
    sampleCount = int(len(input_data))    #12
    
    # mean center correction
    for i in range(input_data.shape[0]):
        input_data[i, :] -= input_data[i, :].mean()   # the mean of every row becomes zero
        
    # Get the reference spectrum. If not given, estimate it from the mean    
    # Define a new array and populate it with the corrected data
    data_msc = np.zeros_like(input_data)
    
    for i in range(input_data.shape[0]):  #12
        for j in range(0, sampleCount, 6):  # why is it a step of 10?
            #print(j+6)
            ref.append(np.mean(input_data[j:j + 6], axis=0))  #mean spectra of the first 10 or 2 last spectra in the dictionary
            # Run regression
            fit = np.polyfit(ref[i], input_data[i, :], 1, full=True)
            # Apply correction
            data_msc[i, :] = (input_data[i, :] - fit[0][1]) / fit[0][0]

    return (data_msc)


def MakeMSCDict(abs_dict):
    msc_dict = {}
    for key in abs_dict:
        msc_dict[key] = msc(abs_dict[key])
    return msc_dict

def ProcessMeanSpectrawithMSC(mean_spectra_dict):
    processed_array = MakeMSCDict(ConverttoAbsDict(SavGolDict(mean_spectra_dict)))
    return processed_array

def PlotMSCSpectra(mean_spectra_dict, ww):
    
    for key in mean_spectra_dict:
        counter = 1
        plt.figure(figsize=(10, 10))
        for spectra in range(len(mean_spectra_dict[key])):
            plt.plot(ww, mean_spectra_dict[key][spectra], label=f'{counter}')
            counter+= 1
        #plt.title(f"Mean absorption spectra for each of the 12 areas of brick {key} after MSC", fontsize=20)    
        plt.legend(title='Area', title_fontsize=17, fontsize = 13)
        plt.xlabel('Wavelength [nm]', fontsize = 17)
        plt.ylabel('Absorption', fontsize = 17)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
       # plt.savefig(f'{key}_dried_MSC_spectra.png')
        

        
if __name__=="__main__":

    #%% Check MSC function
    header_path = r'D:\untreated\Gruppe 2 Spektrum'
    headers = CollectOnlyHeaders(header_path)
    ww_nm = Roundww(Wavelengths(headers, header_path))
    raw_spectra = OpenDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\Resultater 2\Raw Spectra 1103', headers, '_spectra')
    
    msc_spectra = ProcessMeanSpectrawithMSC(raw_spectra)     
    PlotMSCSpectra(msc_spectra, ww_nm)      
   
    #%%
    # running MSC on all untreated spectra
    header_path = r'D:\untreated\Gruppe 2 Spektrum'
    headers = CollectOnlyHeaders(header_path)
    ww_nm = Roundww(Wavelengths(headers, header_path))
    spectra = OpenDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\Resultater 2\Raw Spectra 1103', headers, '_spectra')
    msc_spectra = ProcessMeanSpectrawithMSC(spectra)
    name_ending = '_msc_spectra'
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v3 edge and mortar removal\MSC spectra', msc_spectra, name_ending)
    group_nr = 9
    subgroup=''
    SpectraToDF(msc_spectra, ww_nm, group_nr).to_excel('G'+str(group_nr)+subgroup+name_ending+'.xlsx')
    
    # Join the excelfiles
    joined_df = JoinExcelFiles(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v3 edge and mortar removal\MSC spectra\excel')
    joined_df.to_excel('All Bricks Mean Spectra after MSC.xlsx')

   

    ##############################################################
    # Kjøre MSC på alle tørka spektere
    header_path = r'D:\torre\Gruppe 1 ABCD\D'
    headers = CollectOnlyHeaders(header_path)
    ww_nm = Roundww(Wavelengths(headers, header_path))
    spectra = OpenDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\Resultater 2\dried spectra\dried mean spectra', headers, '_spectra')
    msc_spectra = ProcessMeanSpectrawithMSC(spectra)
    name_ending = '_msc_spectra'
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v3 edge and mortar removal\MSC spectra dried', msc_spectra, name_ending)
    group_nr = 1
    subgroup='D'
    SpectraToDF(msc_spectra, ww_nm, group_nr).to_excel('G'+str(group_nr)+subgroup+name_ending+'.xlsx')
    
    # Join the excelfiles
    joined_df = JoinExcelFiles(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v3 edge and mortar removal\MSC spectra dried\excel')
    joined_df.to_excel('All Dried Bricks Mean Spectra after MSC.xlsx')