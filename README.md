# Preprocessing and analysis of hyperspectral images of brick

This repository contains the files necessary for the pre-processing of brick images. In addition, example files regarding the explorative analysis of the bricks' physical and spectral data are added.

The following files describe the pre-processing of the hyperspectral images:
- ImportImagesFunctions.py
- SaveLoadDictionaries.py
- PreprocessingFunctions.py
- MakeBloksFunctions.py
- StreamlineFunction.py
- SNVProcessingFunctions.py
- MSCProcessingFunctions.py
- MortarRemovalFunctions

The file "StreamlineFunction.py" handles the entire processing from import of images to generation of mean spectra. However, the preprocessing of the generated spectra are executed through "SNVProcessingFunctions.py" or "MSCProcessingFunctions.py".

Testing and acquisition of the mortar training data are included in the file "Mortar Removal Testing both Models". It includes testing of both the K-means clustering model and PLS-DA model. The datasets with the mortar and non-mortar spectral samples have been uploaded to the Teams-group of the Master Thesis project.

Examples regarding the use of the functions are added under the __name__=="__main__" in each file.

## Code for exploratory analyse of bricks' physical and spectral properties
 - Correlation matrix - Physical Properties.ipynb
 - Explorative analysis - Physical Properties.ipynb
 - Explorative analysis - Dried brick spectra.ipynb
   
## Assignment of normally distributed values to the spectra
- Make dataset with norm.dist. values.ipynb

## Classification based on spectral properties
- Classification based on spectral properties.ipynb

## Example of prediction model development
- Example Prediction - Predicting FR.ipynb

Note: the files are added as examples. Code which was developed, but not used, is excluded. 
