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
        if 'Fall.mat' in file:
            file_path = os.path.join(subdir, file)
            # Navigate up two levels to get to the desired folder
            session_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
            filesNeuro.append((file_path, session_name))

nFiles = len(filesAqua)
print(f"Number of suite2p files = {nFiles}")


#%% Load the data from the csv file

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
features = 

#%% Categorize/bin dff by regions

nRegions = np.shape(regionIdx)[0]
nEvents = np.shape(regionIdx)[1]
nFrames = np.shape(dffMat)[1]
nBins = 10 # Number of frames to bin


evtCount = np.zeros([nRegions,nFrames]) # nRegions by nFrames to store 
evtBinned = np.zeros([nRegions,nFrames-nBins])

for region in range(nRegions):
    evtDffCluster = evtDff[~np.isnan(regionIdx[region,:])] # extract dff peaks from one region
    evtFramesCluster = evtFrames[~np.isnan(regionIdx[region,:])] # extract dff frame idx from one region
    
    # Now fill in evtCount with each region's dff of their own events
    for y in range(len(evtFramesCluster)):
        evtCount[region,int(evtFramesCluster[y])] = evtCount[region,int(evtFramesCluster[y])] + evtDffCluster[y]

    for yy in range(evtCount.shape[1]-nBins):
        evtBinned[region,yy] = np.sum(evtCount[region,yy+nBins])
    
    evtBinned[region,yy+1:nBins] = np.sum(evtCount[region,yy+1:])

#%% Visualize each region's events

# Function to compute DFF
def computeDFF(data):
    baselineFrames = 48  # Number of frames to use for baseline calculation

    baseline = np.mean(data[:, :baselineFrames], axis=1, keepdims=True)
    dff = (data - baseline) / baseline

    return dff

dff = computeDFF(evtCount)

plt.figure(figsize=(20,8))

for region in range(nRegions):
    
    plt.plot(range(nFrames),dff[region,:], label=f'Region {region + 1}')

plt.title(f"Astrocyte DFF by region \n{session_name}", fontsize=15)
plt.legend()
sns.despine()