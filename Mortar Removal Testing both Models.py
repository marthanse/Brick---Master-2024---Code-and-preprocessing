# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:45:01 2024

Testing edge removal of brick outline and removal of mortar

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



#%% Testing removal of mortar by kmeans

header_folder = r'D:\untreated\Gruppe 1 ABCD\A'
headers = CollectOnlyHeaders(header_folder)
image_folder = r'D:\Whitecorrected images\untreated'
#image_files, headers = CollectImages(folder)
#images = ImportImages(folder, image_files, headers)

# Test on already whitecorrected images
images = OpenDictNumpy(image_folder, headers, '_whitecorrected_image')

#%%Creating model based on A5A
A5A = images['A5_A']
A5A = CutWhiteRef(A5A)

rows_A5A, cols_A5A, bands_A5A = A5A.shape
tabular_data_A5A = A5A.reshape((rows_A5A*cols_A5A, bands_A5A))
tab_A5A = pd.DataFrame(tabular_data_A5A)
tab_A5A = tab_A5A.interpolate()

#%%  Trying PCA
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#%%
pca_A5A = PCA(n_components=5)
pca_model_A5A = pca_A5A.fit(tab_A5A)
data_pca_A5A = pca_model_A5A.transform(tab_A5A)
#data_pca = pca.fit_transform(tab_A5A)

#%%
# Clustering with KMeans

kmeans_model_A5A = KMeans(n_clusters=5, random_state=0).fit(data_pca_A5A)

labels_from_A5A = kmeans_model_A5A.labels_

# Reshape back to the original image shape
A5A_data_reshaped = labels_from_A5A.reshape((rows_A5A, cols_A5A))

#%% 
plt.figure()
plt.imshow(A5A_data_reshaped)
plt.show()

# her er den lyseste grønne steinen

#%% Create model from A8A

A8A = images['A8_A']
A8A = CutWhiteRef(A8A)

rows_A8A, cols_A8A, bands_A8A = A8A.shape
tabular_data_A8A = A8A.reshape((rows_A8A*cols_A8A, bands_A8A))
tab_A8A = pd.DataFrame(tabular_data_A8A)
tab_A8A = tab_A8A.interpolate()

pca_A8A = PCA(n_components=5)
pca_model_A8A = pca_A8A.fit(tab_A8A)
#data_pca = pca2.fit_transform(tab_A8A)
data_pca_A8A = pca_model_A8A.transform(tab_A8A)

#%%
# Clustering with KMeans

kmeans_model_A8A = KMeans(n_clusters=5, random_state=0).fit(data_pca_A8A)

labels_from_A8A = kmeans_model_A8A.labels_

# Reshape back to the original image shape
A8A_data_reshaped = labels_from_A8A.reshape((rows_A8A, cols_A8A))

#%% 
plt.figure()
plt.imshow(A8A_data_reshaped)
plt.show()

# den mørkeste grønne er murstein

#%% Apply the PCA and kmeans of A8A to A5A

transformed_A5A = pca_model_A8A.transform(tab_A5A)
pred_labels_A5A = kmeans_model_A8A.predict(transformed_A5A)

A5A_pred = pred_labels_A5A.reshape((rows_A5A, cols_A5A))

#%%
plt.figure()
plt.imshow(A5A_pred)
plt.show()

# her er den lyseste grønne stein wtf?

#%% Apply the PCA and kmeans of A5A to A8A

transformed_A8A = pca_model_A5A.transform(tab_A8A)
pred_labels_A8A = kmeans_model_A5A.predict(transformed_A8A)

A8A_pred = pred_labels_A8A.reshape((rows_A8A, cols_A8A))

#%%
plt.figure()
plt.imshow(A8A_pred)
plt.show()

# A5A did not work well on A8A
# but A8A work to some degree on A5A

#%% Testing A8A model applied to a brick without mortar

A9A = images['A9_A']
A9A = CutWhiteRef(A9A)

rows_A9A, cols_A9A, bands_A9A = A9A.shape
tabular_data_A9A = A9A.reshape((rows_A9A*cols_A9A, bands_A9A))
tab_A9A = pd.DataFrame(tabular_data_A9A)
tab_A9A = tab_A9A.interpolate()

#%%
transformed_A9A = pca_model_A8A.transform(tab_A9A)
pred_labels_A9A = kmeans_model_A8A.predict(transformed_A9A)

A9A_pred = pred_labels_A9A.reshape((rows_A9A, cols_A9A))

plt.figure()
plt.imshow(A9A_pred)
plt.show()


#%% Find out which group is the yellow one

A9A_pred_1 = A9A_pred[:,:]==4
plt.figure()
plt.imshow(A9A_pred_1)
plt.show()

# --> can use this to retrieve filter for each brick!

#%% Make mortar filter for entire A-group
# Apply A8A model to alle bricks in the A group
images = CutAllWhiteRef(images)
#%%

transformed_images = {}
for key in images:
    image = images[key]
    rows, cols, bands = image.shape
    tab = pd.DataFrame(image.reshape((rows*cols, bands))).interpolate()
    transformed_img = pca_model_A8A.transform(tab)
    pred_labels = kmeans_model_A8A.predict(transformed_img)
    image_pred = pred_labels.reshape((rows, cols))
    transformed_images[key] = image_pred
    
    plt.figure()
    plt.imshow(image_pred)
    plt.show()
    
#%%
for key in transformed_images:
    plt.figure()
    plt.imshow(transformed_images[key])
    plt.title(key)
    plt.show()   
    
#%% Get the light green areas of every brick

for key in transformed_images:
    check = (transformed_images[key][:,:] == 2) | (transformed_images[key][:,:] == 3)
    plt.figure()
    plt.imshow(check)
    plt.title(key)
    plt.show()
    
    
#%% Train on A7_B, see if the one mortar lump can be grouped out
A7B = images['A7_B']
A7B = CutWhiteRef(A7B)

rows_A7B, cols_A7B, bands_A7B = A7B.shape
tabular_data_A7B = A7B.reshape((rows_A7B*cols_A7B, bands_A7B))
tab_A7B = pd.DataFrame(tabular_data_A7B)
tab_A7B = tab_A7B.interpolate()    

pca_A7B = PCA(n_components=5)
pca_model_A7B = pca_A7B.fit(tab_A7B)
#data_pca = pca2.fit_transform(tab_A8A)
data_pca_A7B = pca_model_A7B.transform(tab_A7B)

kmeans_model_A7B = KMeans(n_clusters=5, random_state=0).fit(data_pca_A7B)

labels_from_A7B = kmeans_model_A7B.labels_

# Reshape back to the original image shape
A7B_data_reshaped = labels_from_A7B.reshape((rows_A7B, cols_A7B))

plt.figure()
plt.imshow(A7B_data_reshaped)
plt.show()

# her er det gule de ubrukelige, tørre delene av steinen. Kan ta med penetrasjonsdybde i diskusjon.

#%% Applier A7B modellen til alle steinene
images = CutAllWhiteRef(images)
transformed_images = {}
for key in images:
    image = images[key]
    rows, cols, bands = image.shape
    tab = pd.DataFrame(image.reshape((rows*cols, bands))).interpolate()
    transformed_img = pca_model_A7B.transform(tab)
    pred_labels = kmeans_model_A7B.predict(transformed_img)
    image_pred = pred_labels.reshape((rows, cols))
    transformed_images[key] = image_pred
    
    plt.figure()
    plt.imshow(image_pred, cmap='tab10')
    plt.show()
    
# funker bra for noen, men ikke for andre


#%% Testing PLS-DA and training model for mortar or no mortar!

# Try first on BB Standard group where there is only mortar in the holes
header_folder = r'D:\untreated\Gruppe 5 BB Standard'
headers = CollectOnlyHeaders(header_folder)
image_folder = r'D:\Whitecorrected images\untreated'

# Test on already whitecorrected images
images = OpenDictNumpy(image_folder, headers, '_whitecorrected_image')
images = CutAllWhiteRef(images)

#%% Look at BB Standard

ShowImages(images)

#%%

ww = Roundww(Wavelengths(headers, header_folder))
#%%
pixel_list = [images['BB11_A'][439:469,276:300],
              images['BB11_A'][565:580,83:107],
              images['BB11_B'][439:469,276:300],
              images['BB11_B'][565:580,83:107],
              images['BB12_A'][285:309,155:176],
              images['BB12_A'][757:790,249:285],
              images['BB12_B'][130:160,83:110],
              images['BB12_B'][613:630,173:210],
              images['BB13_A'][113:137,161:195],
              images['BB13_B'][697:721,152:176],
              images['BB13_B'][118:145, 258:285],
              images['BB16_A'][343:364, 152:188],
              images['BB16_A'][116:146, 149:185],
              images['BB16_B'][845:894, 207:280],
              images['BB16_B'][117:138, 161:191],
              images['BB19_A'][116:143, 270:304],
              images['BB19_A'][786:801, 252:273],
              images['BB19_B'][313:356, 15:76],
              images['BB19_B'][122:156, 277:328]]

nr_samples = len(pixel_list)
mortar = np.zeros(nr_samples*288).reshape(nr_samples,288)
       
for i in range(0, nr_samples):
    tempim = pixel_list[i].mean(axis=0)
    mortar[i,:]=tempim.mean(axis=0)

#%% Make the mortar into a dataframe

label_mortar = [1]*nr_samples
label_nomortar = [0]*nr_samples
# Make the dataframe
df_mortar = pd.DataFrame(mortar, columns =[np.arange(len(ww))], dtype = float)
df_mortar["mortar"] = label_mortar
df_mortar["no mortar"] = label_nomortar
df_mortar.to_csv("mortar", encoding='utf-8', index=False)

#%% Selecting pixels that are not mortar

no_mortar_list = [images['BB11_A'][487:550, 321:364],
              images['BB11_A'][611:650, 131:222],
              images['BB11_A'][771:804, 167:198],
              images['BB11_A'][1012:1064, 222:270],
              
              images['BB11_B'][1000:1042, 98:190],
              images['BB11_B'][100:131, 273:297],
              images['BB11_B'][85:137, 318:370],
              images['BB11_B'][780:820, 327:373],
    
              images['BB12_A'][597:627, 143:179],
              images['BB12_A'][540:585, 312:367],
              images['BB12_A'][863:908, 137:305],
              
              images['BB12_B'][818:850, 213:249],
              images['BB12_B'][977:1000, 137:180],
              
              images['BB13_A'][430:490, 4:60],
              
              images['BB16_A'][846:890, 128:210],
              images['BB16_A'][158:186, 198:240],
              
              images['BB16_B'][22:74, 304:368],
              images['BB16_B'][504:540, 270:360],
              
              images['BB19_A'][829:900, 10:60]]

nr_samples = len(no_mortar_list)
no_mortar = np.zeros(nr_samples*288).reshape(nr_samples,288)
       
for i in range(0, nr_samples):
    tempim = no_mortar_list[i].mean(axis=0)
    no_mortar[i,:]=tempim.mean(axis=0)

#%% Make the no mortar into a dataframe

label_mortar = [0]*nr_samples
label_nomortar = [1]*nr_samples

# Make the dataframe
df_no_mortar = pd.DataFrame(no_mortar, columns =[np.arange(len(ww))], dtype = float)
df_no_mortar["no mortar"] = label_nomortar
df_no_mortar["mortar"] = label_mortar
df_no_mortar.to_csv("no mortar", encoding='utf-8', index=False)


#%% Combine the csv files into one dataframe

mortar_4_data = pd.read_csv("mortar 4")
nomortar_4_data = pd.read_csv("no mortar 4")
mortar_5_data = pd.read_csv("mortar 5")
nomortar_5_data = pd.read_csv("no mortar 5")
mortar_9_data = pd.read_csv("mortar 9")
nomortar_9_data = pd.read_csv("no mortar 9")
mortar_7_data = pd.read_csv("mortar 7 updated")
nomortar_7_data = pd.read_csv("no mortar 7")
mortar_2_data = pd.read_csv("mortar 2")
nomortar_2_data = pd.read_csv("no mortar 2")
nomortar_3_data = pd.read_csv("no mortar 3")

df = pd.concat([mortar_4_data, nomortar_4_data, mortar_5_data, nomortar_5_data,
                mortar_9_data, nomortar_9_data, mortar_7_data, nomortar_7_data, 
                mortar_2_data, nomortar_2_data, nomortar_3_data], ignore_index=True)
#%% 

from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
#%%
X_train = df.iloc[:, 0:288]
y_train = df.iloc[:, 288:290]

#%% Getting ww to the columns of the dataframe
header_folder = r'D:\untreated\Gruppe 5 BB Standard'
headers = CollectOnlyHeaders(header_folder)
ww = Roundww(Wavelengths(headers, header_folder))
#%% Standardizing
sc = StandardScaler()
df_sc = pd.DataFrame(sc.fit_transform(X_train),columns = ww)

#%%
pls = PLSRegression(n_components=5)
pls.fit(X_train, y_train)

#%% Predict image pixels based on pls model

image_folder = r'D:\Whitecorrected images\untreated'
# Test on already whitecorrected images
images = OpenDictNumpy(image_folder, headers, '_whitecorrected_image')
images = CutAllWhiteRef(images)
#%%
# PLS modellen har blitt trent på individuelle mean spektre, 1x288 må mates inn i modellen
# og predictes label på den pikselen
# kommer til å bli veldig computationally demanding to go through each pixel??
ex_img = images['BB11_A']
rows = ex_img.shape[0]
cols = ex_img.shape[1]
bands = ex_img.shape[2]

ex_labels = np.empty([rows, cols])
for i in range(rows):
    for j in range(cols):
        values = pls.predict(ex_img[i, j].reshape(1,-1))
        if values[0, 0]>values[0, 1]:  # values[0] is mortar, values[1] is no mortar
            ex_labels[i, j] = 1  # it is mortar
        else:
            ex_labels[i, j] = 0  # it is not mortar

#test = pls.predict(ex_img[50, 220].reshape(1,-1))

#%% Show the real image
ShowImage(ex_img)
#%% Shpw the labels after predictions
plt.figure()
plt.imshow(ex_labels)
plt.show()

# BB11_A funket dette dritbra på!!!
# BB13_B funker også bra!
#%% Prøv på en til i samme gruppe!
ex_img = images['BB13_B']
rows = ex_img.shape[0]
cols = ex_img.shape[1]
bands = ex_img.shape[2]

ex_labels = np.empty([rows, cols])
for i in range(rows):
    for j in range(cols):
        if not np.isnan(ex_img[i,j]).any():
            values = pls.predict(ex_img[i, j].reshape(1,-1))
            if values[0, 0]>values[0, 1]:  # values[0] is mortar, values[1] is no mortar
                ex_labels[i, j] = 1  # it is mortar
            else:
                ex_labels[i, j] = 0  # it is not mortar
        else: 
            ex_labels[i, j] = 1  # assume it is weird an remove it anyway

#%% Apply this model to BB Stor
BBstor_folder = r'D:\untreated\Gruppe 4 BB Stor'
BB_stor_headers= CollectOnlyHeaders(BBstor_folder)
BB_stor = OpenDictNumpy(r'D:\Whitecorrected images\untreated', BB_stor_headers, '_whitecorrected_image')
BB_stor = CutAllWhiteRef(BB_stor)
#%% Test på BB18_A
ex_img = BB_stor['BB18_A']
rows = ex_img.shape[0]
cols = ex_img.shape[1]
bands = ex_img.shape[2]


ex_labels = np.empty([rows, cols])
for i in range(rows):
    for j in range(cols):
        if not np.isnan(ex_img[i,j]).any():
            values = pls.predict(ex_img[i, j].reshape(1,-1))
            if values[0, 0]>values[0, 1]:  # values[0] is mortar, values[1] is no mortar
                ex_labels[i, j] = 1  # it is mortar
            else:
                ex_labels[i, j] = 0  # it is not mortar
        else: 
            ex_labels[i, j] = 1  # assume it is weird an remove it anyway

#%% Test på BB20_A - masse mørtel

ex_img = BB_stor['BB20_A']
rows = ex_img.shape[0]
cols = ex_img.shape[1]
bands = ex_img.shape[2]

ex_labels = np.empty([rows, cols])
for i in range(rows):
    for j in range(cols):
        if not np.isnan(ex_img[i,j]).any():
            values = pls.predict(ex_img[i, j].reshape(1,-1))
            if values[0, 0]>values[0, 1]:  # values[0] is mortar, values[1] is no mortar
                ex_labels[i, j] = 1  # it is mortar
            else:
                ex_labels[i, j] = 0  # it is not mortar
        else: 
            ex_labels[i, j] = 1  # assume it is weird an remove it anyway
            
#%% Teste en bakside med tynt lag av mørtel/puss

ex_img = BB_stor['BB21_B']
rows = ex_img.shape[0]
cols = ex_img.shape[1]
bands = ex_img.shape[2]

ex_labels = np.empty([rows, cols])
for i in range(rows):
    for j in range(cols):
        if not np.isnan(ex_img[i,j]).any():
            values = pls.predict(ex_img[i, j].reshape(1,-1))
            if values[0, 0]>values[0, 1]:  # values[0] is mortar, values[1] is no mortar
                ex_labels[i, j] = 1  # it is mortar
            else:
                ex_labels[i, j] = 0  # it is not mortar
        else: 
            ex_labels[i, j] = 1  # assume it is weird an remove it anyway
            
#%%
ShowImages(BB_stor)            
#%% Legge til treningsdata for flere grupper
# BB Stor

mortar_BB_stor = [BB_stor['BB18_A'][879:900, 189:230],
                  BB_stor['BB18_B'][660:716, 28:40],
                  BB_stor['BB20_A'][710:740, 212:250],
                  BB_stor['BB20_B'][108:148, 90:135],
                  BB_stor['BB23_B'][869:894, 115:180]]

nr_samples = len(mortar_BB_stor)
mortar = np.zeros(nr_samples*288).reshape(nr_samples,288)
#label_mortar = [0]*nr_samples
       
for i in range(0, nr_samples):
    tempim = mortar_BB_stor[i].mean(axis=0)
    mortar[i,:]=tempim.mean(axis=0)

#%% Save the mortar data for group 4

label_mortar = [1]*nr_samples  # her har det skjedd noe feil, dette er mørtel gruppa, skal ha 1 i merke
label_nomortar = [0]*nr_samples

df_mortar = pd.DataFrame(mortar, columns =[np.arange(len(ww))], dtype = float)
df_mortar["no mortar"] = label_nomortar
df_mortar["mortar"] = label_mortar
df_mortar.to_csv("mortar 4", encoding='utf-8', index=False)
#%%
no_mortar_BB_stor = [BB_stor['BB18_A'][57:82, 251:290],
                     BB_stor['BB21_A'][886:917, 98:160],
                     BB_stor['BB18_B'][857:905, 28:107],
                     BB_stor['BB20_B'][305:335, 268:340],
                     BB_stor['BB21_B'][18:69, 317:370],
                     BB_stor['BB21_B'][666:710, 20:70]
    ]

nr_samples = len(no_mortar_BB_stor)
no_mortar = np.zeros(nr_samples*288).reshape(nr_samples,288)
       
for i in range(0, nr_samples):
    tempim = no_mortar_BB_stor[i].mean(axis=0)
    no_mortar[i,:]=tempim.mean(axis=0)

#label_nomortar = [1]*nr_samples
    
#%% Save the no mortar data for group 4

label_mortar = [0]*nr_samples
label_nomortar = [1]*nr_samples

# Make the dataframe
df_no_mortar = pd.DataFrame(no_mortar, columns =[np.arange(len(ww))], dtype = float)
df_no_mortar["no mortar"] = label_nomortar
df_no_mortar["mortar"] = label_mortar
df_no_mortar.to_csv("no mortar 4", encoding='utf-8', index=False)

#%% Get pixels from Group 9

fornebu_folder = r'D:\untreated\Gruppe 9 Fornebu'
fornebu_headers= CollectOnlyHeaders(fornebu_folder)
fornebu = OpenDictNumpy(r'D:\Whitecorrected images\untreated', fornebu_headers, '_whitecorrected_image')
fornebu = CutAllWhiteRef(fornebu)

#%%

mortar_fornebu = [fornebu['FV1_A'][71:107, 183:233],
          fornebu['FV1_A'][539:634, 168:203],
          fornebu['FV2_A'][9:56, 191:270],
          fornebu['FV3_A'][804:866, 183:242],
          fornebu['FV1_B'][249:301, 145:241],
          fornebu['FV5_B'][22:57, 144:244]     
          ]

nr_samples = len(mortar_fornebu)
mortar = np.zeros(nr_samples*288).reshape(nr_samples,288)
#label_mortar = [0]*nr_samples
       
for i in range(0, nr_samples):
    tempim = mortar_fornebu[i].mean(axis=0)
    mortar[i,:]=tempim.mean(axis=0)
    
#%%
ww = Roundww(Wavelengths(headers, folder))
#%% Save the mortar from group 9

label_mortar = [1]*nr_samples
label_nomortar = [0]*nr_samples

df_mortar = pd.DataFrame(mortar, columns =[np.arange(len(ww))], dtype = float)
df_mortar["no mortar"] = label_nomortar
df_mortar["mortar"] = label_mortar
df_mortar.to_csv("mortar 9", encoding='utf-8', index=False)

#%% Get no mortar samples from group 9
no_mortar_fornebu = [fornebu['FV1_A'][655:722, 340:370],
                     fornebu['FV3_A'][154:207, 236:259],
                     fornebu['FV5_A'][323:359, 132:165],
                     fornebu['FV1_B'][450:494, 10:43],
                     fornebu['FV6_B'][857:884, 257:320],
                     fornebu['FV6_A'][326:365, 123:171]                     
                     ]

nr_samples = len(no_mortar_fornebu)
no_mortar = np.zeros(nr_samples*288).reshape(nr_samples,288)
       
for i in range(0, nr_samples):
    tempim = no_mortar_fornebu[i].mean(axis=0)
    no_mortar[i,:]=tempim.mean(axis=0)
    
#%% Save the no mortar group 9

label_mortar = [0]*nr_samples
label_nomortar = [1]*nr_samples

# Make the dataframe
df_no_mortar = pd.DataFrame(no_mortar, columns =[np.arange(len(ww))], dtype = float)
df_no_mortar["no mortar"] = label_nomortar
df_no_mortar["mortar"] = label_mortar
df_no_mortar.to_csv("no mortar 9", encoding='utf-8', index=False)

#%% Function mortar detection

def MortarDetection(image):
    
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
                labels[i, j] = 1  # assume it is weird an remove it anyway
    final_labels = labels
    return final_labels
#%% Test new trained model on Fornebu images

ex_img = BB_stor['BB23_A']
labeled = MortarDetection(ex_img)

#%%
"""
Det kan virke som om labels er snudd på hodet for fornebu gruppen. 
Kan det samme ha skjedd med gruppe 5 og 4?
"""

#%% 
ShowImage(ex_img)
plt.figure()
plt.imshow(labeled)
plt.show()

#%%
ex_img = BB_stor['BB20_A']
labeled_BB20A = MortarDetection(ex_img)

#%% Test på andre grupper, eks gruppe 1 A


folder = r'D:\untreated\Gruppe 7 Saga'
headers= CollectOnlyHeaders(folder)
images = OpenDictNumpy(r'D:\Whitecorrected images\untreated', headers, '_whitecorrected_image')
images = CutAllWhiteRef(images)
#%%
ex_img = images['SAGA1_A']
labeled = MortarDetection(ex_img)

#%% Utvid treningssett med ikke mørtel fra saga
ShowImages(images)

#%%
no_mortar_saga = [images['SAGA1_A'][630:680, 168:250],
                     images['SAGA2_A'][528:570, 151:240],
                     images['SAGA4_A'][121:170, 159:260],
                     images['SAGA2_B'][388:423, 157:205],
                     images['SAGA2_B'][758:808, 43:113]
                     ]

nr_samples = len(no_mortar_saga)
no_mortar = np.zeros(nr_samples*288).reshape(nr_samples,288)
       
for i in range(0, nr_samples):
    tempim = no_mortar_saga[i].mean(axis=0)
    no_mortar[i,:]=tempim.mean(axis=0)
    
#%% Save no mortar saga
label_mortar = [0]*nr_samples
label_nomortar = [1]*nr_samples

# Make the dataframe
df_no_mortar = pd.DataFrame(no_mortar, columns =[np.arange(len(ww))], dtype = float)
df_no_mortar["no mortar"] = label_nomortar
df_no_mortar["mortar"] = label_mortar
df_no_mortar.to_csv("no mortar 7", encoding='utf-8', index=False)

#%% Mortar saga
mortar_saga = [images['SAGA3_B'][837:911, 259:338],
          images['SAGA2_B'][103:130, 17:55],
          images['SAGA3_A'][88:135, 273:332]
         ]

nr_samples = len(mortar_saga)
mortar = np.zeros(nr_samples*288).reshape(nr_samples,288)
#label_mortar = [0]*nr_samples
       
for i in range(0, nr_samples):
    tempim = mortar_saga[i].mean(axis=0)
    mortar[i,:]=tempim.mean(axis=0)

#%% Save mortar saga griup 7

label_mortar = [1]*nr_samples
label_nomortar = [0]*nr_samples

df_mortar = pd.DataFrame(mortar, columns =[np.arange(len(ww))], dtype = float)
df_mortar["no mortar"] = label_nomortar
df_mortar["mortar"] = label_mortar
df_mortar.to_csv("mortar 7 updated", encoding='utf-8', index=False)

#%% Teste på andre grupper
folder = r'D:\untreated\Gruppe 2 Spektrum'
headers= CollectOnlyHeaders(folder)
images = OpenDictNumpy(r'D:\Whitecorrected images\untreated', headers, '_whitecorrected_image')
images = CutAllWhiteRef(images)
#%%
ex_img = fornebu['FV5_B']
labeled = MortarDetection(ex_img)
#%%
ShowImage(ex_img)
plt.figure()
plt.imshow(labeled)
plt.show()

#%%
ShowImages(images)

#%% No mortar spektrum

no_mortar_sp = [images['SP5_A'][29:80, 120:200],
                     images['SP5_A'][577:625, 317:354],
                     images['SP1_A'][124:203, 6:40],
                     images['SP3_B'][29:64, 284:358]
                     ]

nr_samples = len(no_mortar_sp)
no_mortar = np.zeros(nr_samples*288).reshape(nr_samples,288)
       
for i in range(0, nr_samples):
    tempim = no_mortar_sp[i].mean(axis=0)
    no_mortar[i,:]=tempim.mean(axis=0)
    
#%% Save no mortar spektrum
label_mortar = [0]*nr_samples
label_nomortar = [1]*nr_samples

# Make the dataframe
df_no_mortar = pd.DataFrame(no_mortar, columns =[np.arange(len(ww))], dtype = float)
df_no_mortar["no mortar"] = label_nomortar
df_no_mortar["mortar"] = label_mortar
df_no_mortar.to_csv("no mortar 2", encoding='utf-8', index=False)

#%% Mortar spektrum
mortar_sp = [images['SP2AB_B'][823:860, 248:340],
          images['SP4_A'][193:242, 278:320]
         ]

nr_samples = len(mortar_sp)
mortar = np.zeros(nr_samples*288).reshape(nr_samples,288)
#label_mortar = [0]*nr_samples
       
for i in range(0, nr_samples):
    tempim = mortar_sp[i].mean(axis=0)
    mortar[i,:]=tempim.mean(axis=0)
#%% Save mortar spektrum
label_mortar = [1]*nr_samples
label_nomortar = [0]*nr_samples

df_mortar = pd.DataFrame(mortar, columns =[np.arange(len(ww))], dtype = float)
df_mortar["no mortar"] = label_nomortar
df_mortar["mortar"] = label_mortar
df_mortar.to_csv("mortar 2", encoding='utf-8', index=False)

#%% Testing on the rest: Moss, M, ABCD
folder = r'D:\untreated\Gruppe 3 Moss'
headers= CollectOnlyHeaders(folder)
images = OpenDictNumpy(r'D:\Whitecorrected images\untreated', headers, '_whitecorrected_image')
images = CutAllWhiteRef(images)

#%% no mortar Moss
no_mortar_moss = [images['moss1_A'][378:447, 13:61],
                     images['moss1_B'][410:450, 305:360],
                     images['moss1_B'][780:812, 15:83]
                     ]

nr_samples = len(no_mortar_moss)
no_mortar = np.zeros(nr_samples*288).reshape(nr_samples,288)
       
for i in range(0, nr_samples):
    tempim = no_mortar_moss[i].mean(axis=0)
    no_mortar[i,:]=tempim.mean(axis=0)
#%%
label_mortar = [0]*nr_samples
label_nomortar = [1]*nr_samples

# Make the dataframe
df_no_mortar = pd.DataFrame(no_mortar, columns =[np.arange(len(ww))], dtype = float)
df_no_mortar["no mortar"] = label_nomortar
df_no_mortar["mortar"] = label_mortar
df_no_mortar.to_csv("no mortar 3", encoding='utf-8', index=False)
