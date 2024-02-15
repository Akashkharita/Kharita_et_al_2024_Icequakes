
# Discrimination between icequakes and earthquakes in southern Alaska: an exploration of waveform features using random forest algorithm

---
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This github repository contains all the jupyter notebooks for the submitted work about **"Discrimination between icequakes and earthquakes in southern Alaska: an exploration of waveform features using random forest algorithm"**
The manuscript is submitted to Geophysical Journal International (GJI). 


### Creating the environment
To run the jupyter notebooks, one has to create an environment using 

```
conda env create -f iq_vs_tq_environment.yml
```

### Objective - 
1) To assess the machine learning models to automatically discriminate icequakes and earthquakes in the southern Alaska.
2) To explore the performance of different feature space. 
3) To explore the transferability of machine learning model. 

### Data 
Waveforms corresponding to all the icequakes between 2005 and 2022 and located within 50 km radius of the Columbia glacier available in the USGS ANSS catalog was downloaded at 15 stations located within 100 km of the Columbia glacier. This resulted in around 2650 icequakes. An earthquake catalog between the magnitudes of 0 to 3 and depth range of 0 to 100 km was also downloaded. To maintain the class balance, only the latest 3000 earthquakes were selected. The **[catalog parameters](https://github.com/Akashkharita/Kharita_et_al_2024_Icequakes/blob/23469bf6cf58d8992332d892c13eddef4dcaf445/Catalogs/catalog_parameters.txt)** are present in this file  As the stations were deployed at different dates and for different amount of duration, the data availability varies from station to station. Overall number of waveforms equal to 43k. The data should be downloaded from here before running jupyter notebooks - [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7523349.svg)](https://doi.org/10.5281/zenodo.7523349)

### Method 

All the waveforms are detrended, tapered (by 10%) and filtered between (0.5-25 Hz) followed by removal of instrument response. Features are then extracted from the processed waveforms using TSFEL feature extraction library for various duration between 15, 30, 45s, 1 and 2 minutes. The duration was computed using obspy.taup program with iaspei-91 velocity model. Different sort of features (statistical, temporal, and spectral features were extracted) and their performances was compared. All the code for extracting features can be found in  - **[tsfel_feature_extraction.py](https://github.com/Akashkharita/Icequake_Earthquake_Discrimination/blob/main/tsfel_raw_data_feature_extraction.py)**. The folder **[tsfel_features](https://github.com/Akashkharita/Icequake_Earthquake_Discrimination/tree/main/tsfel_features)** contains all types of extracted features. 



### Naming convention of waveforms

1) I have named the downloaded icequakes and earthquakes for each station in a very specific way that helps me in processing. 
2) A typical icequake waveform is named icequake(its number on catalog)_(station name).mseed
   For example - icequake which appears at 244th in the USGS event catalog and recorded at stationn "SCM" will be named as icequake244_SCM.mseed
3) Same naming convention applies for the earthquake waveforms

 
### Description of tasks of Jupyter Notebooks

**[Final_features_performance_comparisons.ipynb](https://github.com/Akashkharita/Icequake_Earthquake_Discrimination/blob/main/Final_feature_performance_comparisons.ipynb)**
This notebook shows the comparison of performances (accuracy, sensitivity and specificity) with statistical, temporal, spectral and AFS features with imbalanced and balanced classes. 

**[Final_transfer_learning](https://github.com/Akashkharita/Icequake_Earthquake_Discrimination/blob/main/Final_Transfer_Learning.ipynb)**
This notebook shows the performance of a machine learning model trained on one station and tested on the another using "all" features for 1 minute duration of the waveforms. Further it contains codes for showing the waveforms and spectrograms for stations in each group. 

**[Final duration_testing](https://github.com/Akashkharita/Icequake_Earthquake_Discrimination/blob/main/Final_duration_testing.ipynb)**
This notebook shows the performance with different durations (15, 30, 45s, 1min and 2minutes). 30s appear to produce best results, Performance in general decreases with increasing duration, however there isnt any significant difference in the results. 

**[Final_missclassified_events](https://github.com/Akashkharita/Icequake_Earthquake_Discrimination/blob/main/Final_misclassified_events.ipynb))**
This notebook contains the code to plot the missclassified event along with the prediction probabilities. Two type of plots - first plot is of missclassified events only, second plot contains all the plots with prediction probabilities less than 0.5. 



