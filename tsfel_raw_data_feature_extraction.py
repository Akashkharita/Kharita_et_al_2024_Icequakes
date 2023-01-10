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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score
#from tsfresh import extract_features
#from tsfresh.feature_extraction import EfficientFCParameters, MinimalFCParameters
import time
import tsfel
import warnings
warnings.filterwarnings("ignore")


from obspy.clients.fdsn import Client
from obspy.geodetics.base import gps2dist_azimuth
from obspy.taup import TauPyModel
model = TauPyModel(model="iasp91")


iq_lats = pd.read_csv('icequakes_catalog.csv')['latitude']
iq_lons = pd.read_csv('icequakes_catalog.csv')['longitude']

eq_lats = pd.read_csv('earthquakes_catalog.csv')['latitude']
eq_lons = pd.read_csv('earthquakes_catalog.csv')['longitude']

stns = pd.read_csv('gmap-stations.txt', sep='|', skiprows=[2,6])


stations = stns.values[:,1].astype('str')
stn_lats = stns.values[:,2].astype('float')
stn_lons = stns.values[:,3].astype('float')
gl_lat, gl_lon = 61.219722, -146.895278





## loading the raw icequake and earthquake waveforms from a given station for a given duration and frequency band
def load_raw_data(station, dur = 1, freq_band = None):
    
        ## defining a variable to store 
        X1 = []
        for i in tqdm(range(len(glob('Data/icequake_waveforms/*'+station+'*')))):
            st = obspy.read(glob('Data/icequake_waveforms/*'+station+'*')[i])
            st.detrend()
            if freq_band:
                st.filter(type='bandpass', freqmin=freq_band[0], freqmax=freq_band[1])

            d = st.select(channel='BHZ')[0]
            c = Client('IRIS')
            if len(d) == 15000:
                network = d.stats.network
                station = d.stats.station
                location = d.stats.location
                channel = d.stats.channel
                starttime = d.stats.starttime
                endtime = d.stats.endtime
                sr = d.stats.sampling_rate #storing sampling rate
                inv = c.get_stations(network = network, station= station, location= location, channel=channel,
                         starttime=starttime, endtime=endtime, level="response")
   
                d.remove_response(inventory= inv, output = "VEL")
    
                string = glob('Data/icequake_waveforms/*'+station+'*')[i]
        
                index = int(string.split('/')[2].split('icequake')[1].split('_')[0]) # we are trying to find the position of this waveform in the USGS catalog 
            # So that we can find its position in USGS catalog. Once we find its position in the catalog
            # We can extract event lat, lon, depth and origin time. This information is going to be useful for obspy.taup. 
    

                val = pd.read_csv('icequakes_catalog.csv').values[index]
                time, lat, lon, depth = val[0], val[1], val[2], val[3]
                stns = pd.read_csv('gmap-stations.txt', sep='|', skiprows=[2,6])
                stns.index = stns[' Station ']
                stn_lat = stns.at[station, ' Latitude ']
                stn_lon = stns.at[station, ' Longitude ']


                dist = gps2dist_azimuth(lat, lon, stn_lat, stn_lon)[0]/(111*1000)
                arrivals = model.get_travel_times(source_depth_in_km=depth, distance_in_degree=dist)
                arr = arrivals[0].time


                X1.append(d[int(arr*sr):int(arr*sr+dur*60*sr)])
                        ## X1 is storing the normalized fft of all the icequakes. \n",

        X2 = []
        for i in tqdm(range(len(glob('Data/earthquake_waveforms/*'+station+'*')))):
            st = obspy.read(glob('Data/earthquake_waveforms/*'+station+'*')[i])
            st.detrend()
            if freq_band:
                st.filter(type='bandpass', freqmin=freq_band[0], freqmax=freq_band[1])
           
            d = st.select(channel='BHZ')[0]
            if len(d) == 15000:
                network = d.stats.network
                station = d.stats.station
                location = d.stats.location
                channel = d.stats.channel
                starttime = d.stats.starttime
                endtime = d.stats.endtime
                inv = c.get_stations(network = network, station= station, location= location, channel=channel,
                         starttime=starttime, endtime=endtime, level="response")
    
                d.remove_response(inventory= inv, output = "VEL")
        
                string = glob('Data/earthquake_waveforms/*'+station+'*')[i]

                index = int(string.split('/')[2].split('earthquake')[1].split('_')[1])
                val = pd.read_csv('earthquakes_catalog.csv').values[index]
                time, lat, lon, depth = val[0], val[1], val[2], val[3]
                stns = pd.read_csv('gmap-stations.txt', sep='|', skiprows=[2,6])
                stns.index = stns[' Station ']
                stn_lat = stns.at[station, ' Latitude ']
                stn_lon = stns.at[station, ' Longitude ']

                dist = gps2dist_azimuth(lat, lon, stn_lat, stn_lon)[0]/(111*1000)
                arrivals = model.get_travel_times(source_depth_in_km=depth, distance_in_degree=dist)
                arr = arrivals[0].time                                                                                           
                X2.append(d[int(arr*sr):int(arr*sr+dur*60*sr)])
                    ## X1 is storing the normalized fft of all the icequakes. \n",
        
        X = X1 + X2
        X = np.array(X)   ## concatenating the icequake and earthquake arrays.\n",
        a = np.ones(len(X1))  ## defining the labels; 1 for icequakes and 0 for earthquakes\n",
        b = np.zeros(len(X2))
        c = np.concatenate([a,b])
        y = c        # label vector\n",
       
   
        return X,y



def extract_tsfel_features(X, dur = 1, domain = 'statistical'):
       
        
        # If no argument is passed retrieves all available features   \n",
       
        if domain == 'all':
 
            cfg_file = tsfel.get_features_by_domain(domain = None)
        else:
            cfg_file = tsfel.get_features_by_domain(domain) 
        
        X_train = []
        for i in range(len(X)):
            X_train.append(tsfel.time_series_features_extractor(cfg_file, X[i], fs=50, window_size=int(dur*60*50), 
                                                                n_jobs = -1, verbose = 1))
            
       
        feature_labels = X_train[0].columns.values
        
        X_features = np.reshape(X_train, [len(X), len(feature_labels)])
    
        return X_features, feature_labels




    
## 1 min
#starttime = time.time()
for i in tqdm(range(len(stations))):

            X,y = load_raw_data(stations[i], dur =0.5, freq_band = None)
            X_f, f_labels = extract_tsfel_features(X, dur = 0.5, domain = 'all')
            np.savetxt("/home/ak287/Icequakes_vs_Tectonicquakes/tsfel_raw_data_features/all/30s/"+stations[i]+".txt", X_f)
            np.savetxt("/home/ak287/Icequakes_vs_Tectonicquakes/tsfel_raw_data_features/all/30s/"+stations[i]+"_labels.txt", y)

   
         

#endtime = time.time()
#np.savetxt("/home/ak287/Icequakes_vs_Tectonicquakes/tsfel_raw_data_features/all/30s/time_taken.txt", np.array(endtime-starttime))
