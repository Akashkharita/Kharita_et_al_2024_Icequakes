## importing necessary dependies

import obspy 
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from tqdm import tqdm
import os
import time
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from obspy.geodetics.base import gps2dist_azimuth

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from sklearn.feature_selection import RFE

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

#from tsfresh import extract_features
#from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters
import time
import pygmt
import matplotlib
import random



iq_lats = pd.read_csv('icequakes_catalog.csv')['latitude']
iq_lons = pd.read_csv('icequakes_catalog.csv')['longitude']

eq_lats = pd.read_csv('earthquakes_catalog.csv')['latitude']
eq_lons = pd.read_csv('earthquakes_catalog.csv')['longitude']

stns = pd.read_csv('gmap-stations.txt', sep='|', skiprows=[2,6])


stations = stns.values[:,1].astype('str')
stn_lats = stns.values[:,2].astype('float')
stn_lons = stns.values[:,3].astype('float')
gl_lat, gl_lon = 61.219722, -146.895278

no_of_icequakes = []
no_of_earthquakes = []
for i in range(15):
    no_of_icequakes.append(len(glob('Data/icequake_waveforms/*'+stations[i]+'*')))
    no_of_earthquakes.append(len(glob('Data/earthquake_waveforms/*'+stations[i]+'*')))
    
    
total_events = np.array(no_of_icequakes)+np.array(no_of_earthquakes)
order = np.argsort(total_events)




def compute_dist_snr_az(station, l=2):
    """"
    
    This function will compute the SNR of each waveform at a given station and stored it in an array 
    
    :station = station name
    :l = [0,1,2] for [E,N,Z]

    
    """
    
    snr = []
    dist = []    ## container where SNRs will be stored. 
    az = []
    Mag = []
    for i in tqdm(range(len(glob('Data/icequake_waveforms/*'+station+'*')))):
        st = obspy.read(glob('Data/icequake_waveforms/*'+station+'*')[i]).select(channel='BHZ')[0]
        # loading the waveform whose SNR is to be computed
        string = glob('Data/icequake_waveforms/*'+station+'*')[i]
        
        if len(st.data) == 15000:

            index = int(string.split('/')[2].split('icequake')[1].split('_')[0]) # we are trying to find the position of this waveform in the USGS catalog 
            # So that we can find its position in USGS catalog. Once we find its position in the catalog
            # We can extract event lat, lon, depth and origin time. This information is going to be useful for obspy.taup. 
    
    
            val = pd.read_csv('icequakes_catalog.csv').values[index]
            lat, lon, mag =  val[1], val[2], val[4]
            stns = pd.read_csv('gmap-stations.txt', sep='|', skiprows=[2,6])
            stns.index = stns[' Station ']
            stn_lat = stns.at[station, ' Latitude ']
            stn_lon = stns.at[station, ' Longitude ']

            dist.append(gps2dist_azimuth(lat, lon, stn_lat, stn_lon)[0]/(1000))
            az.append(gps2dist_azimuth(lat, lon, stn_lat, stn_lon)[1])
            
            st.detrend()
            st.taper(0.1)
            st.filter('bandpass', freqmin = 0.5, freqmax = 25)
            
            
            snr.append(np.max(abs(st.data))/np.std(abs(st.data)))
            Mag.append(mag)
            
    for i in tqdm(range(len(glob('Data/earthquake_waveforms/*'+station+'*')))):
        st = obspy.read(glob('Data/earthquake_waveforms/*'+station+'*')[i]).select(channel='BHZ')[0]
        string = glob('Data/earthquake_waveforms/*'+station+'*')[i]
        
        if len(st.data) == 15000:
            index = int(string.split('/')[2].split('earthquake')[1].split('_')[1])
            val = pd.read_csv('earthquakes_catalog.csv').values[index]
            time, lat, lon, depth, mag = val[0], val[1], val[2], val[3], val[4]
            stns = pd.read_csv('gmap-stations.txt', sep='|', skiprows=[2,6])
            stns.index = stns[' Station ']
            stn_lat = stns.at[station, ' Latitude ']
            stn_lon = stns.at[station, ' Longitude ']

            dist.append(gps2dist_azimuth(lat, lon, stn_lat, stn_lon)[0]/(1000))
            az.append(gps2dist_azimuth(lat, lon, stn_lat, stn_lon)[1])
            
            st.detrend()
            st.taper(0.1)
            st.filter('bandpass', freqmin = 0.5, freqmax = 25)
            
            snr.append(np.max(abs(st.data))/np.std(abs(st.data)))
            Mag.append(mag)
    return dist, snr, az, Mag




common = []
dist = []
snr = []
az = []
mag = []
for i in tqdm(range(len(stations))):
    common = compute_dist_snr_az(stations[i])
    dist.append(common[0])
    snr.append(common[1])
    az.append(common[2])
    mag.append(common[3])
    
    

## writing the computed distances, snr and azimuths, so when you use it again, you dont have to compute from 
## previous cell, simply load them and use. 

for i in range(len(stations)):
    np.savetxt("event_parameters/"+stations[i]+"_dist.txt", np.array(dist[i]))
    np.savetxt("event_parameters/"+stations[i]+"_snr.txt", np.array(snr[i]))
    np.savetxt("event_parameters/"+stations[i]+"_az.txt", np.array(az[i]))
    np.savetxt("event_parameters/"+stations[i]+"_mag.txt", np.array(mag[i]))
    
    
   