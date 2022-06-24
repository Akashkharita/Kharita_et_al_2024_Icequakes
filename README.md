# Icequake_Earthquake_Discrimination
Discrimination between Icequakes and Earthquakes in Southeast Alaska 

### Necessary Dependencies

1) Numpy
2) Matplotlib
3) Obspy
4) Scipy
5) Scikit-Learn
6) tqdm
7) tsfresh


### Naming convention of waveforms

1) I have named the downloaded icequakes and earthquakes for each station in a very specific way that helps me in processing. 
2) A typical icequake waveform is named icequake(its number on catalog)_(station name).mseed
   For example - icequake which appears at 244th in the USGS event catalog and recorded at stationn "SCM" will be named as icequake244_SCM.mseed
3) Same naming convention applies for the earthquake waveforms

 
### Description of tasks of Jupyter Notebooks


**AK_Tsfresh_vs_AFS_features.ipynb** - This jupyter notebook shows the feature extraction process. 784 statistical features were extracted from each waveform (5 minutes in duration) at each station. Since it is a time consuming process, it is wise to write these features on the disk. Similarly, Fourier Amplitude Spectrum (FAS) of each waveform was computed and saved.Then the random forest was applied on both sets of features individually at different stations. 50 iterations were performed to compute the mean accuracy and the performance for both sets of features was compared. It was found that FAS outperform features extracted using Tsfresh. 

