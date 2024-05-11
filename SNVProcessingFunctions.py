# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 11:07:07 2024

File for functions going to be applied to spectra
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
#%% Functions

def Roundww(wavelengths):
    ww_nm = []
    for i in range(len(wavelengths)):
        ww_nm.append(round(wavelengths[i]))
    return ww_nm


def SpectraToDF(spectra, ww_nm, group_nr):
        
    df = pd.DataFrame(columns=(ww_nm + ['group', 'brick_id']))
    for key in spectra:
        #spectra_arrays = []
        spectra_arrays = spectra[key] 
        brick_id_times_12 = [key]*12
        group_id = [group_nr]*12
        df2 = pd.DataFrame(spectra_arrays, columns=ww_nm)
        df2['group'] = group_id
        df2['brick_id'] = brick_id_times_12
        joined_dataframe = pd.concat([df, df2], ignore_index=True)
        
        df = joined_dataframe
        
    return df
            
def JoinExcelFiles(folder_path):
    # Assumes excel files with equal column names
    df1 = pd.DataFrame()  # create empty dataframe
    for file in os.listdir(folder_path):
        if file.endswith(".xlsx"):
            df2 = pd.read_excel(folder_path+'\\'+file)
            joined_dataframe = pd.concat([df1, df2], ignore_index=True)
            df1 = joined_dataframe
    df1 = df1.iloc[:, 1:]  # remove  empty column
    return df1

def SavitskyGolay(mean_specter):
    """

    Parameters
    ----------
    mean_specter : the mean specter of the brick, array float32
    5 : window_length
    1 : polyorder
    axis : The axis of the array x along which the filter is to be applied. 
            The default is 1.

    Returns
    -------
    None.

    """
    sbrick = []
    for spectre in mean_specter:
        
        sbrick.append(savgol_filter(spectre, 5, 1, axis=-1))

    return sbrick

def SavGolDict(mean_specter_dict):
    savgol = {}
    for key in mean_specter_dict:
        savgol[key] = SavitskyGolay(mean_specter_dict[key])
    return savgol

def ConverttoAbsorption(sbrick):
    #x = swood.shape
    
    absbrick = []
    
    for spectre in sbrick:
        absbrick.append(np.log10(1 /(spectre+0.001)))
    
    return absbrick

def ConverttoAbsDict(savgol_dict):
    abs_dict = {}
    for key in savgol_dict:
        abs_dict[key] = ConverttoAbsorption(savgol_dict[key])
    return abs_dict

def snv(abs_array):
    
    snv_array = np.empty(len(abs_array))
    #snv_array = abs_array.copy()

    for i in range(len(abs_array)):
        snv_array[i] = (abs_array[i] - np.mean(abs_array)) / (np.std(abs_array))
            
        if math.isnan(snv_array[i]):
            snv_array[i]=0
            
    return snv_array

def snvAllarrays(abs_12_spectra):
    snv_result = []
    for s in abs_12_spectra:
        snv_result.append(snv(s))
    return snv_result

def MakeSNVdict(abs_dict):
    snv_dict = {}
    for key in abs_dict:
        snv_dict[key] = snvAllarrays(abs_dict[key])
    return snv_dict

def ProcessMeanSpectra(mean_spectra_dict):
    processed_array = MakeSNVdict(ConverttoAbsDict(SavGolDict(mean_spectra_dict)))
   # savgol_dict = SavGolDict(mean_spectra_dict)
   # abs_dict = ConverttoAbsDict(savgol_arrays)
    #snv_arrays = snvAllarrays(abs_array)
    return processed_array

def SaveandProcessSpectra(path_for_saving, spectra_for_processing_path, header_path, name_ending='_snv_spectra', group_nr=0, subgroup=''): 
    # name_ending: '_spectra' etc
    """
    spectra_for_processing: a dictionary with unprocessed spectra retrieved from brick sections
    
    group_nr: equals to 0 by default, but are supposed to be between 1 and 9
    """
    headers = CollectOnlyHeaders(header_path)
    ww_nm = Roundww(Wavelengths(headers, header_path))
    spectra = OpenDictNumpy(spectra_for_processing_path, headers, '_spectra')
    processed_spectra = ProcessMeanSpectra(spectra)
    SaveDictNumpy(path_for_saving, processed_spectra, name_ending)
    SpectraToDF(processed_spectra, ww_nm, group_nr).to_excel('G'+str(group_nr)+subgroup+name_ending+'.xlsx')


if __name__=="__main__":
    
    
    folder = r'D:\untreated\Gruppe 2 Spektrum'
    headers = CollectOnlyHeaders(folder)
    # Import the spectra to be processed
    spectra = OpenDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\Resultater 2\Raw Spectra 1103', headers, '_spectra')
    # Process the spectra for all steps included SNV
    processed_spectra = ProcessMeanSpectra(spectra)
    ww_nm = Roundww(Wavelengths(headers, folder))
    PlotSNVSpectra(processed_spectra, ww_nm)
    
    # Get absorption spectra
    abs_spectra = ConverttoAbsDict(SavGolDict(spectra))
    PlotAbsSpectra(abs_spectra, ww_nm)
    

    # save to excel
    SpectraToDF(processed_spectra, ww_nm, 2).to_excel('G2_snv_spectra_mortar_removed.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v3 edge and mortar removal\spectra after mortar removal\dried snv spectra', processed_spectra, '_snv_spectra')
    
    ##### Joining files
    joined_df = JoinExcelFiles(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v3 edge and mortar removal\spectra after mortar removal\snv spectra\excel')
    joined_df.to_excel('All bricks no mortar SNV.xlsx')    ##########################################################################
    
    
    #Testing streamline function
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 1 ABCD\A')
    spectra = OpenDictNumpy(r'D:\brick outlines 2902\G1 ABCD', headers, '_spectra')
    processed_spectra = ProcessMeanSpectra(spectra)
    
    SaveandProcessSpectra(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2',
                          r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\Raw mean spectra\Mean spectra data - np and xlsx\1 Spectra - Untreated, before SG',
                          r'D:\untreated\Gruppe 1 ABCD\A',
                          name_ending='_snv_spectra',
                          group_nr=1,
                          subgroup='A')
    

    ##########################################################################
    # Testing the snv function
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 1 ABCD\A')
    
    ww = Wavelengths(headers, r'D:\untreated\Gruppe 1 ABCD\A')
    ww_nm = Roundww(ww)
    
    spectra = OpenDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\Abs data', headers, '_abs_spectra')
    snv_array = snvAllarrays(spectra['A4_A'])
    snv_dict = MakeSNVdict(spectra)
    
    # Apply SNV to all absoprtion spectra
    # G1 A
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 1 ABCD\A')
    spectra = OpenDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\Abs data', headers, '_abs_spectra')
    snv_spectra = MakeSNVdict(spectra)
    SpectraToDF(snv_spectra, ww_nm, 1).to_excel('G1A_snv_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SNV data', snv_spectra, '_snv_spectra')
   
    #G1 B
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 1 ABCD\B')
    spectra = OpenDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\Abs data', headers, '_abs_spectra')
    snv_spectra = MakeSNVdict(spectra)
    SpectraToDF(snv_spectra, ww_nm, 1).to_excel('G1B_snv_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SNV data', snv_spectra, '_snv_spectra')
    
    # G1C
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 1 ABCD\C')
    spectra = OpenDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\Abs data', headers, '_abs_spectra')
    snv_spectra = MakeSNVdict(spectra)
    SpectraToDF(snv_spectra, ww_nm, 1).to_excel('G1C_snv_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SNV data', snv_spectra, '_snv_spectra')
    
    #G1D
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 1 ABCD\D')
    spectra = OpenDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\Abs data', headers, '_abs_spectra')
    snv_spectra = MakeSNVdict(spectra)
    SpectraToDF(snv_spectra, ww_nm, 1).to_excel('G1D_snv_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SNV data', snv_spectra, '_snv_spectra')
    
    # G2
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 2 Spektrum')
    spectra = OpenDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\Abs data', headers, '_abs_spectra')
    snv_spectra = MakeSNVdict(spectra)
    SpectraToDF(snv_spectra, ww_nm, 2).to_excel('G2_snv_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SNV data', snv_spectra, '_snv_spectra')
    
    # G3
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 3 Moss')
    spectra = OpenDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\Abs data', headers, '_abs_spectra')
    snv_spectra = MakeSNVdict(spectra)
    SpectraToDF(snv_spectra, ww_nm, 3).to_excel('G3_snv_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SNV data', snv_spectra, '_snv_spectra')
    
    #G4
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 4 BB Stor')
    spectra = OpenDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\Abs data', headers, '_abs_spectra')
    snv_spectra = MakeSNVdict(spectra)
    SpectraToDF(snv_spectra, ww_nm, 4).to_excel('G4_snv_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SNV data', snv_spectra, '_snv_spectra')

    #G5
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 5 BB Standard')
    spectra = OpenDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\Abs data', headers, '_abs_spectra')
    snv_spectra = MakeSNVdict(spectra)
    SpectraToDF(snv_spectra, ww_nm, 5).to_excel('G5_snv_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SNV data', snv_spectra, '_snv_spectra')
    
    #G6
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 6 M')
    spectra = OpenDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\Abs data', headers, '_abs_spectra')
    snv_spectra = MakeSNVdict(spectra)
    SpectraToDF(snv_spectra, ww_nm, 6).to_excel('G6_snv_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SNV data', snv_spectra, '_snv_spectra')
    
    #G7
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 7 Saga')
    spectra = OpenDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\Abs data', headers, '_abs_spectra')
    snv_spectra = MakeSNVdict(spectra)
    SpectraToDF(snv_spectra, ww_nm, 7).to_excel('G7_snv_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SNV data', snv_spectra, '_snv_spectra')
    
    #G8
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 8 SVA')
    spectra = OpenDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\Abs data', headers, '_abs_spectra')
    snv_spectra = MakeSNVdict(spectra)
    SpectraToDF(snv_spectra, ww_nm, 8).to_excel('G8_SVA_snv_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SNV data', snv_spectra, '_snv_spectra')
    
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 8 SVB')
    spectra = OpenDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\Abs data', headers, '_abs_spectra')
    snv_spectra = MakeSNVdict(spectra)
    SpectraToDF(snv_spectra, ww_nm, 8).to_excel('G8_SVB_snv_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SNV data', snv_spectra, '_snv_spectra')
    
    #G9
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 9 Fornebu')
    spectra = OpenDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\Abs data', headers, '_abs_spectra')
    snv_spectra = MakeSNVdict(spectra)
    SpectraToDF(snv_spectra, ww_nm, 9).to_excel('G9_snv_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SNV data', snv_spectra, '_snv_spectra')
   
    # Join the excelfiles
    joined_df = JoinExcelFiles(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SNV data\excel files')
    
   
    
   ###########################################################################
    
    # Testing av omgjøring til absorbans
    # G1 A
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 1 ABCD\A')
    
    ww = Wavelengths(headers, r'D:\untreated\Gruppe 1 ABCD\A')
    ww_nm = Roundww(ww)
    
    # obs du må hente inn Savitsky-Golay filteret!
    spectra = OpenDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SavGol data', headers, '_sg_spectra')
    abs_spectra = ConverttoAbsDict(spectra)
    SpectraToDF(abs_spectra, ww_nm, 1).to_excel('G1A_abs_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\Abs data', abs_spectra, '_abs_spectra')
    
    #G1 B
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 1 ABCD\B')
    spectra = OpenDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SavGol data', headers, '_sg_spectra')
    abs_spectra = ConverttoAbsDict(spectra)
    SpectraToDF(abs_spectra, ww_nm, 1).to_excel('G1B_abs_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\Abs data', abs_spectra, '_abs_spectra')
    
    # G1C
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 1 ABCD\C')
    spectra = OpenDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SavGol data', headers, '_sg_spectra')
    abs_spectra = ConverttoAbsDict(spectra)
    SpectraToDF(abs_spectra, ww_nm, 1).to_excel('G1C_abs_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\Abs data', abs_spectra, '_abs_spectra')
    
    #G1D
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 1 ABCD\D')
    spectra = OpenDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SavGol data', headers, '_sg_spectra')
    abs_spectra = ConverttoAbsDict(spectra)
    SpectraToDF(abs_spectra, ww_nm, 1).to_excel('G1D_abs_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\Abs data', abs_spectra, '_abs_spectra')
    
    # G2
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 2 Spektrum')
    spectra = OpenDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SavGol data', headers, '_sg_spectra')
    abs_spectra = ConverttoAbsDict(spectra)
    SpectraToDF(abs_spectra, ww_nm, 2).to_excel('G2_abs_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\Abs data', abs_spectra, '_abs_spectra')
    
    # G3
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 3 Moss')
    spectra = OpenDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SavGol data', headers, '_sg_spectra')
    abs_spectra = ConverttoAbsDict(spectra)
    SpectraToDF(abs_spectra, ww_nm, 3).to_excel('G3_abs_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\Abs data', abs_spectra, '_abs_spectra')

    #G4
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 4 BB Stor')
    spectra = OpenDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SavGol data', headers, '_sg_spectra')
    abs_spectra = ConverttoAbsDict(spectra)
    SpectraToDF(abs_spectra, ww_nm, 4).to_excel('G4_abs_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\Abs data', abs_spectra, '_abs_spectra')

    #G5
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 5 BB Standard')
    spectra = OpenDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SavGol data', headers, '_sg_spectra')
    abs_spectra = ConverttoAbsDict(spectra)
    SpectraToDF(abs_spectra, ww_nm, 5).to_excel('G5_abs_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\Abs data', abs_spectra, '_abs_spectra')
    
    #G6
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 6 M')
    spectra = OpenDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SavGol data', headers, '_sg_spectra')
    abs_spectra = ConverttoAbsDict(spectra)
    SpectraToDF(abs_spectra, ww_nm, 6).to_excel('G6_abs_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\Abs data', abs_spectra, '_abs_spectra')
    
    #G7
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 7 Saga')
    spectra = OpenDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SavGol data', headers, '_sg_spectra')
    abs_spectra = ConverttoAbsDict(spectra)
    SpectraToDF(abs_spectra, ww_nm, 7).to_excel('G7_abs_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\Abs data', abs_spectra, '_abs_spectra')
    
    #G8
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 8 SVA')
    spectra = OpenDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SavGol data', headers, '_sg_spectra')
    abs_spectra = ConverttoAbsDict(spectra)
    SpectraToDF(abs_spectra, ww_nm, 8).to_excel('G8_SVA_abs_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\Abs data', abs_spectra, '_abs_spectra')
    
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 8 SVB')
    spectra = OpenDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SavGol data', headers, '_sg_spectra')
    abs_spectra = ConverttoAbsDict(spectra)
    SpectraToDF(abs_spectra, ww_nm, 8).to_excel('G8_SVB_abs_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\Abs data', abs_spectra, '_abs_spectra')
    
    #G9
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 9 Fornebu')
    spectra = OpenDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SavGol data', headers, '_sg_spectra')
    abs_spectra = ConverttoAbsDict(spectra)
    SpectraToDF(abs_spectra, ww_nm, 9).to_excel('G9_abs_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\Abs data', abs_spectra, '_abs_spectra')
    
    # Join the abs excelfiles
    joined_df = JoinExcelFiles(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\Abs data\excel files')
    joined_df.to_excel('All Bricks Mean Spectra Absorption.xlsx')
    
    
    ############################################################################
    # Testing 04.mars av savitsky-golay filter
    G4_headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 4 BB Stor')
    ww = Wavelengths(G4_headers, r'D:\untreated\Gruppe 4 BB Stor')
    ww_nm = Roundww(ww)
    
    G4_spectra = OpenDictNumpy(r'D:\brick outlines 2902\G4 BB Store', G4_headers, '_spectra')

    s = SavitskyGolay(G4_spectra['BB18_A'])
    G4_sg = SavGolDict(G4_spectra)
    SpectraToDF(G4_sg, ww_nm, 4).to_excel('G4_sg_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SavGol data', G4_sg, '_sg_spectra')

    # Lagre resten av steinene med savgol behandling som excel og numpy files
    # obs generelle navn for alle
    # G1
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 1 ABCD\A')
    spectra = OpenDictNumpy(r'D:\brick outlines 2902\G1 ABCD', headers, '_spectra')
    sg = SavGolDict(spectra)
    SpectraToDF(spectra, ww_nm, 1).to_excel('G1A_sg_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SavGol data', sg, '_sg_spectra')
    
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 1 ABCD\B')
    spectra = OpenDictNumpy(r'D:\brick outlines 2902\G1 ABCD', headers, '_spectra')
    sg = SavGolDict(spectra)
    SpectraToDF(spectra, ww_nm, 1).to_excel('G1B_sg_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SavGol data', sg, '_sg_spectra')
    
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 1 ABCD\C')
    spectra = OpenDictNumpy(r'D:\brick outlines 2902\G1 ABCD', headers, '_spectra')
    sg = SavGolDict(spectra)
    SpectraToDF(spectra, ww_nm, 1).to_excel('G1C_sg_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SavGol data', sg, '_sg_spectra')
    
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 1 ABCD\D')
    spectra = OpenDictNumpy(r'D:\brick outlines 2902\G1 ABCD', headers, '_spectra')
    sg = SavGolDict(spectra)
    SpectraToDF(spectra, ww_nm, 1).to_excel('G1D_sg_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SavGol data', sg, '_sg_spectra')
    
    # G2
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 2 Spektrum')
    spectra = OpenDictNumpy(r'D:\brick outlines 2902\G2 Spektrum', headers, '_spectra')
    sg = SavGolDict(spectra)
    SpectraToDF(spectra, ww_nm, 2).to_excel('G2_sg_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SavGol data', sg, '_sg_spectra')
    
    # G3
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 3 Moss')
    spectra = OpenDictNumpy(r'D:\brick outlines 2902\G3 Moss', headers, '_spectra')
    sg = SavGolDict(spectra)
    SpectraToDF(spectra, ww_nm, 3).to_excel('G3_sg_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SavGol data', sg, '_sg_spectra')
    
    #G5
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 5 BB Standard')
    spectra = OpenDictNumpy(r'D:\brick outlines 2902\G5 BB Standard', headers, '_spectra')
    sg = SavGolDict(spectra)
    SpectraToDF(spectra, ww_nm, 5).to_excel('G5_sg_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SavGol data', sg, '_sg_spectra')
    
    #G6
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 6 M')
    spectra = OpenDictNumpy(r'D:\brick outlines 2902\G6 M', headers, '_spectra')
    sg = SavGolDict(spectra)
    SpectraToDF(spectra, ww_nm, 6).to_excel('G6_sg_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SavGol data', sg, '_sg_spectra')
    
    #G7
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 7 Saga')
    spectra = OpenDictNumpy(r'D:\brick outlines 2902\G7 Saga', headers, '_spectra')
    sg = SavGolDict(spectra)
    SpectraToDF(spectra, ww_nm, 7).to_excel('G7_sg_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SavGol data', sg, '_sg_spectra')
    
    #G8
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 8 SVA')
    spectra = OpenDictNumpy(r'D:\brick outlines 2902\G8 SVA', headers, '_spectra')
    sg = SavGolDict(spectra)
    SpectraToDF(spectra, ww_nm, 8).to_excel('G8_SVA_sg_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SavGol data', sg, '_sg_spectra')
    
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 8 SVB')
    spectra = OpenDictNumpy(r'D:\brick outlines 2902\G8 SVB', headers, '_spectra')
    sg = SavGolDict(spectra)
    SpectraToDF(spectra, ww_nm, 8).to_excel('G8_SVB_sg_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SavGol data', sg, '_sg_spectra')
    
    #G9
    headers = CollectOnlyHeaders(r'D:\untreated\Gruppe 9 Fornebu')
    spectra = OpenDictNumpy(r'D:\brick outlines 2902\G9 Fornebu', headers, '_spectra')
    sg = SavGolDict(spectra)
    SpectraToDF(spectra, ww_nm, 9).to_excel('G9_sg_spectra.xlsx')
    SaveDictNumpy(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SavGol data', sg, '_sg_spectra')
    
    # Join the savitsky golay excelfiles
    joined_df = JoinExcelFiles(r'C:\Users\marth\OneDrive\Skole\Master\Kode\murstein-master-2024\Streamline v2\SavGol data\excel files')
    joined_df.to_excel('All Bricks Mean Spectra with SG.xlsx')
    
   
   
    
    
   
