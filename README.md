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

Testing and acquisition of the mortar training data are included in the file "Mortar Removal Testing both Models". It includes testing of both the K-menas clustering model and PLS-DA model. The datasets with the mortar and non-mortar spectral samples have been uploaded to the Teams-group of the Master Thesis project.

Examples regarding the use of the functions are added under the __name__=="__main__" in each file.

## The code for the exploratory analysis of the brick spectra are added in:

The code for the development of the prediction models are:

Note: the files are added as examples, code which were developed and not used are excluded. 
