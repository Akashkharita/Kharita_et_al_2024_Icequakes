
# Icequake_Earthquake_Discrimination
This github repository contains all the jupyter notebooks for the ongoing work about **"Multi Stations Analysis of Icequakes and Earthquakes in Southern Alaska"**. 


### Creating the environment
To run the jupyter notebooks, one has to create an environment using 

> conda env create -f iq_vs_tq_environment.yml

### Objective - 
1) To assess the machine learning models to automatically discriminate icequakes and earthquakes in the southern Alaska.
2) To explore the performance of different feature space. 
3) To explore the transferability of machine learning model. 

### Data - 
Waveforms corresponding to all the icequakes between 2005 and 2022 and located within 50 km radius of the Columbia glacier available in the USGS ANSS catalog was downloaded at 15 stations located within 100 km of the Columbia glacier. This resulted in around 2650 icequakes. An earthquake catalog between the magnitudes of 0 to 3 and depth range of 0 to 100 km was also downloaded. To maintain the class balance, only the latest 3000 earthquakes were selected. As the stations were deployed at different data and for different amount of duration, the data availability varies from station to station. Overall number of waveforms equal to 43k. 

### Method 
We extracted features from the waveforms one minute in duration



### Naming convention of waveforms

1) I have named the downloaded icequakes and earthquakes for each station in a very specific way that helps me in processing. 
2) A typical icequake waveform is named icequake(its number on catalog)_(station name).mseed
   For example - icequake which appears at 244th in the USGS event catalog and recorded at stationn "SCM" will be named as icequake244_SCM.mseed
3) Same naming convention applies for the earthquake waveforms

 
### Description of tasks of Jupyter Notebooks


**AK_Tsfresh_vs_AFS_features.ipynb** - This jupyter notebook shows the feature extraction process. 784 statistical features were extracted from each waveform (5 minutes in duration) at each station. Since it is a time consuming process, it is wise to write these features on the disk. Similarly, Fourier Amplitude Spectrum (FAS) of each waveform was computed and saved.Then the random forest was applied on both sets of features individually at different stations. 50 iterations were performed to compute the mean accuracy and the performance for both sets of features was compared. It was found that FAS outperform features extracted using Tsfresh. 

**AK_Final_Model.ipynb**  - This jupyter notebook contains - (i) Confusion matrix and other metrics for the RF model trained on the combined data from top performing best 7 stations (Multi-station RF). (ii) comparison of f1-scores of RF model on individual stations and multistation RF, (iii) confusion matrix for the individual station and (iv) Geographical Map showing stations color coded and size coded according to accuracies and amount of data. 



