# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:21:31 2024

Save and read in dictionaries
@author: marth
"""

import pickle
import numpy as np

#%% Using pickle

def SaveDict(file_path, dictionary):
    with open(file_path, 'wb') as file:
        pickle.dump(dictionary, file)
        
def LoadDict(file_path):
    with open(file_path, 'rb') as file:
        loaded_dict = pickle.load(file)
    return loaded_dict


#%% lagre med np.save eller np.load isteden

def SaveDictNumpy(file_storage_path, dict_for_saving, type_of_file):
    # type of file (_brick, _filter, _spectra), needs to be string
    for key in dict_for_saving:
        with open(file_storage_path+'\\'+key+type_of_file+'.npy', 'wb') as f:
            #np.save(file_storage_path+'\\'+key, dict_for_saving[key]+'.npy')
            np.save(f, dict_for_saving[key])

def OpenDictNumpy(file_storage_path, headers_dict, type_of_file):
    # type of file (_brick, _filter, _spectra), needs to be string
    new_dict = {}
    for key in headers_dict:
        with open(file_storage_path+'\\'+key+type_of_file+'.npy', 'rb') as f:
            a = np.load(f)
        new_dict[key] = a
    return new_dict
            