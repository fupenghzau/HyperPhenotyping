# -*- coding: utf-8 -*-
"""
Created on Tue May 22 11:17:21 2018

@author: Peng Fu (fupenghzau@gmail.com)
"""

# This file is created to produce smoothed spectral curves

# import modules
import numpy as np
import matplotlib.pyplot as plt
from spectral import open_image
from sklearn import preprocessing
from sklearn.cluster import KMeans
import os
import glob

# may need to be changed based on the file path
date = "2018-06-18"
folder = "D:\Hyperspectral\\2018\\PIKA NIR\\" + date + "\\"
suffix = folder + "*.bil"
filelist = glob.glob(suffix)
filelist.sort()
outfolder = "D:\\Hyperspectral\\2018\PIKA NIR output\\"

GainFile = "D:\\Hyperspectral\\calibration\\GainNIR.txt"
OffsetFile = "D:\\Hyperspectral\\calibration\\OffsetNIR.txt"
WavelengthFile = "D:\\Hyperspectral\\calibration\\wavelengthNIR.txt"
TelfonRefFile = "D:\\Hyperspectral\\calibration\\TelfonInterNIR.txt"

# local functions
wavelength = np.loadtxt(WavelengthFile)
Gain = np.loadtxt(GainFile)
Offset = np.loadtxt(OffsetFile)
TelfonRef = np.loadtxt(TelfonRefFile)  # Reflectance has been interpolated to the wavelength of data cube

for li in range(0, 1):
# local variables

    filename = filelist[9]
    basename = os.path.basename(filename)
    base = os.path.splitext(basename)[0]
    outfilename = outfolder + "\\" + date + "\\" + base + '.txt'
    
    # read header file
    HDRfilename = filename + ".hdr"
    with open(HDRfilename) as f:
        content = f.read().splitlines()
    tempstr = [ti for ti in content if ti.startswith('shutter')][0]
    strlen = len(tempstr)
    DataShutter = float(tempstr[10:strlen])
    ScaleFactor = 8/DataShutter # a scale factor for gain in .ICP
    
    # get the data
    data = open_image(HDRfilename).load()
    msize = data.shape[0]
    nsize = data.shape[1]
    lsize = data.shape[2]
    Radata = np.zeros([msize, nsize, lsize])
    Rkdata = np.zeros([msize*nsize, lsize])
    
    # convert to radiance
    for i in range(0, lsize):
        temp1 = np.reshape(data[:, :, i], [msize, nsize])
        #temp2 = temp1 - np.tile(Offset[i, :], [msize, 1])  # substract dark current
        temp3 = np.multiply(temp1, np.tile(Gain[i, :], [msize, 1])*ScaleFactor)
        Radata[:, :, i] = temp3
        Rkdata[:, i] = np.reshape(temp3, [msize*nsize])
    
    # kmeans
    print("perform kmeans operations......\n")
    numclass = 6
    maximInter = 300
    scaler = preprocessing.StandardScaler().fit(Rkdata)
    Rkdata = scaler.transform(Rkdata)
    classifier = KMeans(n_clusters=numclass, init='k-means++', n_init=10, max_iter=maximInter, tol=0.0001, precompute_distances=True, verbose=0, random_state=None, copy_x=True, n_jobs=1)   
    result = classifier.fit_predict(Rkdata) 
    classimg = result.reshape([msize, nsize] )
    plt.figure(figsize=(10,10))
    plt.imshow(classimg, cmap='Accent')
    plt.colorbar()
    
    
    all_class=[]
    for i in range(numclass):
        classarray=[]
        for j in range(msize):
            for k in range(nsize):
                if(classimg[j,k] == i):
                    classarray.append((j,k))
        all_class.append(classarray)
    
    #calculate the mean spectrum of all pixels at every wavelength for each classes
    class_spectrum_mean=[]
    for p in range (numclass):
        spectrum_sum = 0
        for q in range (len(all_class[p])):# num pixel in class p
            tuplex = all_class[p][q]
            spectrum_sum += 1/(len(all_class[p]))*data[tuplex[0],tuplex[1]]
        class_spectrum_mean.append(spectrum_sum)
    
    #display the spectrum plot for every classes
    plt.figure(figsize=(10,10))
    for h in range(len(class_spectrum_mean)):
        y = class_spectrum_mean[h]
        plt.plot(wavelength, y)
    plt.show()
#    
    # radiance to reflectance based averaged spectrum
    classradiance = np.zeros([numclass, 1])
    for kk in range(0, numclass):
        temp1 = class_spectrum_mean[kk]
        classradiance[kk, 0] = temp1[23]
    
    panelclass = np.where(classradiance==np.max(classradiance))[0][0]
    # when NDVI is positive, choose the radiance that is largest
    classradiance[panelclass, 0] = -10000
    sunlit = np.where(classradiance==np.max(classradiance))[0][0]

    
    tempratio = class_spectrum_mean[sunlit]/class_spectrum_mean[panelclass]
    ARef = np.multiply(tempratio, TelfonRef)*1000
    plt.figure(figsize=(10,10))
    plt.plot(wavelength, ARef)
    plt.show()
    
    # write into txt file
    fid = open(outfilename, 'w')
    for km in range(0, lsize):
        fid.write('%f   %f\n' % (wavelength[km], ARef[km]))
    
    fid.close()
    print("{} is completed\n".format(base))




