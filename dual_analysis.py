# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 12:23:04 2024

For dual imaging analysis

@author: jihop
"""

import numpy as np

import os
import re
import scipy.io 
import pingouin as pg
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import h5py
from scipy.interpolate import interp1d

from sklearn.linear_model import  RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

#%% Define paths and functions

datapath = r'D:\vc-astrocytes\2p3\AstroNeuro03\240527\stim\240527_astroNeuro03_fov4_2x_astro_neuro_puff-000'

#%% Find the _Aqua.mat files and Suite2p mat files

filesAqua = []

for subdir, dirs, files in os.walk(datapath):
    for file in files:
        if '_AQuA.mat' in file:
            file_path = os.path.join(subdir, file)
            # Navigate up two levels to get to the desired folder
            session_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
            filesAqua.append((file_path, session_name))

nFiles = len(filesAqua)
print(f"Number of aqua files = {nFiles}")

filesNeuro = []

for subdir, dirs, files in os.walk(datapath):
    for file in files:
        if 'master.mat' in file:
            file_path = os.path.join(subdir, file)
            # Navigate up two levels to get to the desired folder
            session_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
            filesNeuro.append((file_path, session_name))

nFiles = len(filesAqua)
print(f"Number of suite2p files = {nFiles}")


#%% Load the data

# AQuA data
file_path, session_name = filesAqua[0]

# Load data into a NumPy array in one line
data = h5py.File(file_path, 'r')['res']

# Check the keys (names of datasets or groups) in the file
keys = list(data.keys())
print("Keys in the file:", keys)

features = data['ftsFilter']
dffMat = np.array(data['dffMatFilter']) # Filtered dFF data (events by frames by 2)
evtFrames = np.array(data['ftsFilter']['curve']['dffMaxFrame']).flatten() # frame info (idx) of each event
evtDff = np.array(data['ftsFilter']['curve']['dffMax2']).flatten() # maximum dff of each event 
regionIdx = np.array(data['ftsFilter']['region']['cell']['memberIdx']) # nRegion by nEvents

# Suite2p + behavior data
master = scipy.io.loadmat(filesNeuro[0][0])['master']
dff = master['data'][0][0]['neuro'][0][0]['dff'][0][0]
pupil = master['data'][0][0]['pupil'][0][0][0]['diam'][0]['norm'][0][0][0]
wheel = master['data'][0][0]['wheel'][0][0][0]['norm'][0][0]

# Interpolate wheel and pupil data 
idxOri = np.arange(wheel.size) # Create an array of indices for the original 'wheel' array
idxNew = np.round(np.linspace(0, len(idxOri) - 1, nFrames)).astype(int) # Create an array of indices for the desired number of frames

wheelNew = interp1d(idxOri, wheel)(idxNew) # Use linear interpolation to interpolate 'wheel' data at desired indices
pupilNew = interp1d(idxOri, pupil)(idxNew) # Use linear interpolation to interpolate 'pupil' data at desired indices


#%% Categorize/bin dff by regions

dffByEvt = dffMat[1,:,:].T # nEvents by nFrames
idxEvt = ~np.isnan(regionIdx[:,:])


nRegions = np.shape(regionIdx)[0]
nEvents = np.shape(regionIdx)[1]
nFrames = np.shape(dffMat)[1]
nBins = 10 # Number of frames to bin


evtCount = np.zeros([nRegions,nFrames]) # nRegions by nFrames to store 
evtBinned = np.zeros([nRegions,nFrames-nBins])

# For getting number of events per region over nFrames

for region in range(nRegions):
    evtDffCluster = evtDff[~np.isnan(regionIdx[region,:])] # extract dff peaks from one region
    evtFramesCluster = evtFrames[~np.isnan(regionIdx[region,:])] # extract dff frame idx from one region
    
    # Now fill in evtCount with each region's dff of their own events
    for y in range(len(evtFramesCluster)):
        evtCount[region,int(evtFramesCluster[y])] = evtCount[region,int(evtFramesCluster[y])] + evtDffCluster[y]

    # for yy in range(evtCount.shape[1]-nBins):
    #     evtBinned[region,yy] = np.sum(evtCount[region,yy+nBins])
    
    # evtBinned[region,yy+1:nBins] = np.sum(evtCount[region,yy+1:])
    
#%% Visualize each event's traces 

# For extracting the entire event traces per region over nFrames

dffAll = np.zeros([nRegions,nFrames])

for region in range(nRegions):
    idxRegion = idxEvt[region,:]
    dffRegion = dffByEvt[idxRegion]
    avgDffRegion = np.mean(dffRegion,axis=0)
    dffAll[region,:] = avgDffRegion
    
    plt.figure(figsize=(20,5))
    plt.plot(range(nFrames),np.mean(dffRegion,axis=0),lw=1)
    plt.title(f'Astrocyte #{region+1}')
    sns.despine()
    # for yy in range(dffRegion.shape[0]):
    #     plt.plot(range(nFrames),dffRegion[yy,:])



#%% Visualize each region's events

# import statistics 

# # Function to compute DFF
# def computeDFF(data):
#     baselineFrames = 48  # Number of frames to use for baseline calculation
    
#     # baseline = statistics.mode(data[:,:])
#     baseline = np.mean(data[:, :baselineFrames], axis=1, keepdims=True)
#     dff = (data - baseline) / baseline

#     return dff

# dff = computeDFF(evtCount)

plt.figure(figsize=(20,8))

for region in range(nRegions):
    
    plt.plot(range(nFrames),dff[region,:], label=f'Region {region + 1}')

plt.title(f"Astrocyte DFF by region \n{session_name}", fontsize=15)
plt.legend()
sns.despine()

#%% Find the average DFF trace of all events 

dffAstro = np.mean(dffMat[1,:,:],axis=1)

plt.figure(figsize=(10,5))
plt.plot(avgDff)
sns.despine()

#%% Plot all of the data (neuro, astro, pupil, wheel)

dffNeuro = np.mean(dff, axis=0)

fig, axes = plt.subplots(4,1,figsize=(20,20),sharex=True)

axes[0].plot(dffNeuro)
# sns.heatmap(evtCount,ax=axes[1],cbar=False)
axes[1].plot(np.mean(dffAll,axis=0))
axes[2].plot(pupilNew)
axes[3].plot(wheelNew)

axes[0].set_title('Neuro DFF')
axes[1].set_title('Astro DFF')
axes[2].set_title('Normalized pupil')
axes[3].set_title('Normalized wheel')

# Calculate the number of frames per 30 seconds
frames_per_30sec = 30 * 16

# Generate x-tick positions and labels
x_ticks = np.arange(0, len(dffNeuro), frames_per_30sec)
x_labels = np.arange(0, len(dffNeuro) / 16, 30).astype(int)

# Set the x-ticks and labels
for ax in axes:
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)

# Show the plot
plt.show()

#%% Example GLM 

tr_nIters = 20
resultsDF = pd.DataFrame(columns=['Session','R2'])

# Calculate mean and standard deviation along the frames axis
mean_values = np.mean(dff, axis=1, keepdims=True)
std_values = np.std(dff, axis=1, keepdims=True)

# Calculate z-scores
zDff = (dff - mean_values) / std_values
dfZDff = pd.DataFrame(zDff.T)

X_train = []
sessR2 = []

labelDF = pd.DataFrame()
labelDF['frameID'] = np.arange(0,nFrames,1)

for n in range(nRegions):
    
    sample = evtCount[n,:]
    
    dm = dfZDff
    
    dm['Pupil'] = pupilNew.T
    dm['Wheel'] = wheelNew.T
    
    X_train = []
    R2 = []
    
    nCol = dm.shape[1]
    
    # Use each frame as a trial point and split 
    y = np.arange(nFrames)
    
    for iteration in range(tr_nIters):
        y_train, y_test = train_test_split(y, test_size=0.25, random_state=iteration) # This splits the pupil trace into train and test
        
        X_train = pd.DataFrame(index=range(len(y_train)), columns=range(nCol))
        
        # Use the y_train indices to generate the same frame idx for X_train 
        for i in range(len(y_train)):
            idx1 = y_train[i]
            idx2 = labelDF[labelDF['frameID'] == idx1].frameID
            idx2 = idx2[idx1]
            X_train.iloc[i] = dm.T[idx2]
        
        X_test = pd.DataFrame(index=range(len(y_test)), columns=range(nCol))
        
        for i in range(len(y_test)):
            idx1 = y_test[i]
            idx2 = labelDF[labelDF['frameID'] == idx1].frameID
            idx2 = idx2[idx1]
            X_test.iloc[i] = dm.T[idx2]
        
        for i in range(len(y_train)):
            idx1 = y_train[i]
            idx2 = labelDF[labelDF['frameID'] == idx1].frameID
            idx2 = idx2[idx1]
            hold = sample[idx2]
        
            if i == 0:
                Y_train = hold
            else:
                Y_train = np.hstack((Y_train, hold))
        
        for i in range(len(y_test)):
            idx1 = y_test[i]
            idx2 = labelDF[labelDF['frameID'] == idx1].frameID
            idx2 = idx2[idx1]
            hold = sample[idx2]
        
            if i == 0:
                Y_test = hold
            else:
                Y_test = np.hstack((Y_test, hold))
        
        y_train = Y_train
        y_test = Y_test
        
        glm = RidgeCV(alphas=[0.1, 1.0, 10.0])
        glm.fit(X_train, y_train)
        
        test_r2 = glm.score(X_test, y_test)
        
        R2.append(test_r2)
        
    avgR2 = np.mean(R2)
    
    print(f'For astrocyte {n+1}: Average R2 = {avgR2}')
    
    sessR2.append(avgR2)

sessionDF = pd.DataFrame({'Session':[session],'R2':[sessR2]})
resultsDF = pd.concat([resultsDF,pd.DataFrame({'Session':[session],
                                                'R2':[sessR2]})],ignore_index=True)
