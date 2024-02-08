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
import time
import tsfel
import warnings
warnings.filterwarnings("ignore")


from obspy.clients.fdsn import Client
from obspy.geodetics.base import gps2dist_azimuth
from obspy.taup import TauPyModel
model = TauPyModel(model="iasp91")


import seis_feature


iq_lats = pd.read_csv('/home/ak287/Icequakes_vs_Tectonicquakes/Catalogs/icequakes_catalog.csv')['latitude']
iq_lons = pd.read_csv('/home/ak287/Icequakes_vs_Tectonicquakes/Catalogs/icequakes_catalog.csv')['longitude']

eq_lats = pd.read_csv('/home/ak287/Icequakes_vs_Tectonicquakes/Catalogs/earthquakes_catalog.csv')['latitude']
eq_lons = pd.read_csv('/home/ak287/Icequakes_vs_Tectonicquakes/Catalogs/earthquakes_catalog.csv')['longitude']

stns = pd.read_csv('/home/ak287/Icequakes_vs_Tectonicquakes/Catalogs/gmap-stations.txt', sep='|', skiprows=[2,6])


stations = stns.values[:,1].astype('str')
stn_lats = stns.values[:,2].astype('float')
stn_lons = stns.values[:,3].astype('float')
gl_lat, gl_lon = 61.219722, -146.895278




def load_raw_data(station, dur=1, freq_band=None):
    X1, iq_index = process_waveforms('icequake', station, dur, freq_band)
    X2, eq_index = process_waveforms('earthquake', station, dur, freq_band)

    X = np.array(X1 + X2)
    y = np.concatenate([np.ones(len(X1)), np.zeros(len(X2))])

    return X, y, iq_index, eq_index


def process_waveforms(event_type, station, dur, freq_band):
    X = []
    indices = []

    for i, file_path in enumerate(tqdm(glob(f'/home/ak287/Icequakes_vs_Tectonicquakes/Data/{event_type}_waveforms/*{station}*'))):
        
        # reading the trace data (mseed files)
        st = obspy.read(file_path)

        # detrending
        st.detrend()
        
        # tapering
        st.taper(0.1)

        # applying filter (1-25 Hz) performs good. We will check to see this is consistent for tsfel feature extraction
        # as well. 
        if freq_band:
            st.filter(type='bandpass', freqmin=freq_band[0], freqmax=freq_band[1])

            
        # selecting the z (vertical) component.     
        d = st.select(channel='BHZ')[0]

        
        # selecting only the good data, the original waveforms is of 5 minutes (300x50)
        if len(d) == 15000:
            network, station, location, channel = d.stats.network, d.stats.station, d.stats.location, d.stats.channel
            starttime, endtime = d.stats.starttime, d.stats.endtime
            inv = c.get_stations(network=network, station=station, location=location, channel=channel,
                                 starttime=starttime, endtime=endtime, level="response")

            # removing the instrument response. 
            d.remove_response(inventory=inv, output="VEL")

            
            # getting the index 
            index = get_index(event_type, file_path)
            indices.append(index)

            
            # this is done to compute the distance so we can predict the arrival times. 
            val = pd.read_csv(f'/home/ak287/Icequakes_vs_Tectonicquakes/Catalogs/{event_type}s_catalog.csv').values[index]
            time, lat, lon, depth = val[0], val[1], val[2], val[3]
            stns = pd.read_csv('/home/ak287/Icequakes_vs_Tectonicquakes/Catalogs/gmap-stations.txt', sep='|', skiprows=[2,6])
            stns.index = stns[' Station ']
            stn_lat = stns.at[station, ' Latitude ']
            stn_lon = stns.at[station, ' Longitude ']

            dist = gps2dist_azimuth(lat, lon, stn_lat, stn_lon)[0]/(111*1000)
            arrivals = model.get_travel_times(source_depth_in_km=depth, distance_in_degree=dist)
            arr = arrivals[0].time

            X.append(d[int(arr*50):int(arr*50 + dur*60*50)])

    return X, indices


def get_index(event_type, file_path):
    if event_type == 'icequake':
        index = int(file_path.split('/')[-1].split('icequake')[1].split('_')[0])
    elif event_type == 'earthquake':
        index = int(file_path.split('/')[-1].split('earthquake')[1].split('_')[1])
    else:
        raise ValueError(f"Unknown event type: {event_type}")

    return index



c = Client('IRIS')
for station in stations:
    X,y, iq_index, eq_index  = load_raw_data(station, dur = 1, freq_band = [1, 25])
    total_index = [iq_index+eq_index][0]

    df_features = pd.DataFrame([])
    for i in tqdm(range(len(X))):

            tr = obspy.Trace(X[i])
            tr.stats.sampling_rate = 50
            df = seis_feature.compute_physical_features(tr = tr, envfilter = False)
            df['serial_no'] = total_index[i]
            df['label'] = y[i]
            df_features = pd.concat([df_features, df])

    df_features.to_csv('/home/ak287/Icequakes_vs_Tectonicquakes/Extracted_Features/Physical_features_1_25_'+station+'.csv')