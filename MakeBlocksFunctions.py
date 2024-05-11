# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:35:13 2024

Funksjon for å lage ruter og behandlingen av dem
@author: marth
"""
import numpy as np
import matplotlib.pyplot as plt
from ImportImagesFunctions import ShowImage

#%%

def MakePlotBlockSpectra(image, image_name, wavelengths, background_filter):
    """
    Kilde: emilie sin kode

    Parameters
    ----------
    image : an image
    image_name: the key of the brick, to plot the right title
    wavelengths : list of wavelengths
    background_filter: matches the image in shape. Says which pixels to include or not.

    Returns
    -------
    mean_vals_grid : grid to save the value of each block

    """
    x, y, z = image.shape
    height = 5
    width = 4 # kjører 4 x 3 ruter, 12
    
    height_interval = np.linspace(0, x, height)
    #print(height_interval)
    width_interval = np.linspace(0, y, width)
    #print(width_interval)
    
    mean_vals_grid = []
    #counter = 1
    
    #plt.figure(figsize=(10, 10))
    
    for j in range(len(height_interval) - 1):
        for i in range(len(width_interval) - 1):
            roi = image[round(height_interval[j]):round(height_interval[j + 1]),
                  round(width_interval[i]):round(width_interval[i + 1]), :]  # region of interest
            #ShowImage(roi)
            roi_filter = background_filter[round(height_interval[j]):round(height_interval[j + 1]),
                         round(width_interval[i]):round(width_interval[i + 1])]
            #print(roi_filter)
            #  if roi_filter.all():
            #    print(j,i)
           # plt.figure(figsize=(5, 5))
           # plt.imshow(roi_filter) #Print dark filter to check it looks fine
            #plt.show()  # obs noen av filtrene blir helt blå, fordi alle verdiene er true
            mean_vals = np.zeros(z)
    
            for band in range(z):
               # product = roi[:, :, band]*[roi_filter]
               # print(product)
                mean_vals[band] = np.nanmean(roi[:, :, band]*[roi_filter])
            
            mean_vals_grid.append(mean_vals)
            #plt.plot(wavelengths, mean_vals, label=f'{counter}')
            #counter += 1
    #plt.legend()
    #plt.title(f"Mean spectra for each block in {image_name}")
    
    return mean_vals_grid


def BlockAllBricks(ImageDict, ww, FilterDict):
    BrickswithBlocksDict = {}
    
    for key in ImageDict:
        BrickswithBlocksDict[key] = MakePlotBlockSpectra(ImageDict[key], key, ww, FilterDict[key])
    return BrickswithBlocksDict



#if __name__ == "__main__":    

   
 