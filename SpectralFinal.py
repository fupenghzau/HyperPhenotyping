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

folder = "D:\Hyperspectral\\2017\\PIKA II\\2017-06-20\\" # filename needs to be changed
suffix = folder + "*plot*.bil"
filelist = glob.glob(suffix)
filelist.sort()
outfolder = "D:\\Hyperspectral\\2017\PIKA II output\\"  # folder needs to be changed

for li in range(0, len(filelist)):
# local variables

    filename = filelist[li]
    GainFile = "D:\\Hyperspectral\\calibration\\GainVIS.txt"
    OffsetFile = "D:\\Hyperspectral\\calibration\\OffsetVIS.txt"
    WavelengthFile = "D:\\Hyperspectral\\calibration\\wavelengthVIS.txt"
    TelfonRefFile = "D:\\Hyperspectral\\calibration\\TelfonInter.txt"
    
    
    # local functions
    wavelength = np.loadtxt(WavelengthFile)
    Gain = np.loadtxt(GainFile)
    Offset = np.loadtxt(OffsetFile)
    TelfonRef = np.loadtxt(TelfonRefFile)  # Reflectance has been interpolated to the wavelength of data cube
    basename = os.path.basename(filename)
    base = os.path.splitext(basename)[0]
    outfilename = outfolder + "\\" + filename[30:40] + "\\" + base + '.txt'
    
    # read header file
    HDRfilename = filename + ".hdr"
    with open(HDRfilename) as f:
        content = f.read().splitlines()
    tempstr = [ti for ti in content if ti.startswith('shutter')][0]
    strlen = len(tempstr)
    DataShutter = float(tempstr[10:strlen])
    ScaleFactor = 18/DataShutter # a scale factor for gain in .ICP
    
    # get the data
    data = open_image(HDRfilename).load()
    msize = data.shape[0]
    nsize = data.shape[1]
    lsize = data.shape[2]
    Radata = np.zeros([msize, nsize, lsize])
    Rkdata = np.zeros([msize*nsize, lsize])
    
    # get the dark reference
    dfile = folder + "Dark Correction.bil.hdr"
    Dadata = open_image(dfile).load()
    
    # convert to radiance
    for i in range(0, lsize):
        temp1 = np.reshape(data[:, :, i], [msize, nsize])
        Doffset = np.mean(np.reshape(Dadata[:, :, i], [30, nsize]), axis=0)
        temp2 = temp1 - np.tile(Doffset, [msize, 1])
        #temp2 = temp1 - np.tile(Offset[i, :], [msize, 1])  # substract dark current
        temp3 = np.multiply(temp2, np.tile(Gain[i, :], [msize, 1])*ScaleFactor)
        Radata[:, :, i] = temp3
        Rkdata[:, i] = np.reshape(temp3, [msize*nsize])
    
    # kmeans
    print("perform kmeans operations......\n")
    numclass = 6  # may need to be changed based on the images
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
    
    # radiance to reflectance based averaged spectrum
    classradiance = np.zeros([numclass, 1])
    NDVIclass = np.zeros([numclass, 1])
    for kk in range(0, numclass):
        temp1 = class_spectrum_mean[kk]
        classradiance[kk, 0] = temp1[77]
        NDVIclass[kk, 0] = (temp1[228] - temp1[130])/(temp1[228] + temp1[130])
    
    panelclass = np.where(classradiance==np.max(classradiance))[0][0]
    # when NDVI is positive, choose the radiance that is largest
    sunlit = np.where(NDVIclass==np.max(NDVIclass))[0][0]
    inde = np.where(NDVIclass > 0)[0]
    for tt in range(0, len(inde)):
        tempind = inde[tt]
        if classradiance[tempind, 0] > classradiance[sunlit]:
            sunlit = tempind
    
    tempratio = class_spectrum_mean[sunlit]/class_spectrum_mean[panelclass]
    ARef = np.multiply(tempratio, TelfonRef)*1000
    plt.figure(figsize=(12,10))
    plt.plot(wavelength, ARef)
    plt.show()
    
    # write into txt file
    fid = open(outfilename, 'w')
    for km in range(0, lsize):
        fid.write('%f   %f\n' % (wavelength[km], ARef[km]))
    
    fid.close()
    print("{} is completed\n".format(base))




