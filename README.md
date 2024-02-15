
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
Waveforms corresponding to all the icequakes between 2005 and 2022 and located within 50 km radius of the Columbia glacier available in the USGS ANSS catalog was downloaded at 15 stations located within 100 km of the Columbia glacier. This resulted in around 2650 icequakes **[icequakes catalog](https://github.com/Akashkharita/Kharita_et_al_2024_Icequakes/blob/c402e46c7c38510afa3cb4cac58e4ad8815404bf/Catalogs/icequakes_catalog.csv)**. An earthquake catalog between the magnitudes of 0 to 3 and depth range of 0 to 100 km was also downloaded. To maintain the class balance, only the latest 3000 earthquakes were selected **[earthquakes catalog](https://github.com/Akashkharita/Kharita_et_al_2024_Icequakes/blob/c402e46c7c38510afa3cb4cac58e4ad8815404bf/Catalogs/earthquakes_catalog.csv)**. The **[catalog parameters](https://github.com/Akashkharita/Kharita_et_al_2024_Icequakes/blob/23469bf6cf58d8992332d892c13eddef4dcaf445/Catalogs/catalog_parameters.txt)** are present in this file  as the stations were deployed at different dates and for different amount of duration, the data availability varies from station to station, the station information is present **[here](https://github.com/Akashkharita/Kharita_et_al_2024_Icequakes/blob/c402e46c7c38510afa3cb4cac58e4ad8815404bf/Catalogs/gmap-stations.txt)**. Overall number of waveforms equal to 43k. The data should be downloaded from here before running jupyter notebooks - [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7523349.svg)](https://doi.org/10.5281/zenodo.7523349)

### Method 

All the waveforms are detrended, tapered (by 10%) and filtered between (0.5-25 Hz) followed by removal of instrument response. Features are then extracted from the processed waveforms using TSFEL feature extraction library for various duration between 15, 30, 45s, 1 and 2 minutes. The duration was computed using obspy.taup program with iaspei-91 velocity model. Different sort of features (statistical, temporal, and spectral features were extracted) and their performances was compared. All the code for extracting features can be found in the following scripts - **[tsfel_feature_extraction](https://github.com/Akashkharita/Kharita_et_al_2024_Icequakes/blob/ef3434b0cff4502dc5c8665da577d103909fe251/Feature_Extraction_Scripts/Tsfel_feature_extraction.py)** and **[physical_feature_extraction](https://github.com/Akashkharita/Kharita_et_al_2024_Icequakes/blob/ef3434b0cff4502dc5c8665da577d103909fe251/Feature_Extraction_Scripts/Physical_Feature_Extraction.py)**. The folder **[Extracted_Features](https://github.com/Akashkharita/Kharita_et_al_2024_Icequakes/tree/ef3434b0cff4502dc5c8665da577d103909fe251/Extracted_Features)** contains all types of extracted features for each station. 



### Naming convention of waveforms

1) I have named the downloaded icequakes and earthquakes for each station in a very specific way that helps me in processing. 
2) A typical icequake waveform is named icequake(its number on catalog)_(station name).mseed
   For example - icequake that appears at 244th in the USGS event catalog and is recorded at station "SCM" will be named icequake244_SCM.mseed
3) The same naming convention applies to the earthquake waveforms

 
### Description of tasks of Jupyter Notebooks

**[Feature_Performance_Comparisons.ipynb](https://github.com/Akashkharita/Kharita_et_al_2024_Icequakes/blob/6f6a027adc2b9086d55ee1f3db4b65e53b89284f/Notebooks/Feature_Performance_Comparisons.ipynb)**
This notebook presents a comparison of performance metrics (accuracy, sensitivity, and specificity) using statistical (Tsfel), temporal (Tsfel), spectral (Tsfel), all features from Tsfel, AFS, and physical features in scenarios involving imbalanced and balanced classes. 

**[Ttransfer_Learning](https://github.com/Akashkharita/Kharita_et_al_2024_Icequakes/blob/6f6a027adc2b9086d55ee1f3db4b65e53b89284f/Notebooks/Transfer_Learning.ipynb)**
This notebook shows the performance of a machine learning model trained on one station and tested on another using "all" features for the 1-minute duration of the waveforms. Further, it contains codes for showing the waveforms and spectrograms for stations in each group. 

**[Feature_importances_and_selection](https://github.com/Akashkharita/Kharita_et_al_2024_Icequakes/blob/6f6a027adc2b9086d55ee1f3db4b65e53b89284f/Notebooks/Feature_importances_and_selection.ipynb)**
This notebook show the feature importance in different feature sets at different stations, It also shows the distribution of the top 5 features for different feature sets and different stations. Further it shows the performance variation with the cumulative number of the most important features.  

**[Feature_Computation_Time](https://github.com/Akashkharita/Kharita_et_al_2024_Icequakes/blob/6f6a027adc2b9086d55ee1f3db4b65e53b89284f/Notebooks/Feature_Computation_Time.ipynb)**
This notebook shows the feature computational time for different durations (15, 30, 45s, 1min and 2minutes) of input waveforms and for different set of features.

**[Analysis_of_potentially_mislabeled_events](https://github.com/Akashkharita/Kharita_et_al_2024_Icequakes/blob/6f6a027adc2b9086d55ee1f3db4b65e53b89284f/Notebooks/Analysis_of_potentially_mislabeled_events.ipynb)**
This notebook contains the code for identifying potentially mislabeled events that were consistently misclassified with high probabilities at multiple stations. The code also plots the waveforms and spectrograms of such events. 


**[Hyperparameter_Tuning_and_Comparison_with_other_ML_Models](https://github.com/Akashkharita/Kharita_et_al_2024_Icequakes/blob/6f6a027adc2b9086d55ee1f3db4b65e53b89284f/Notebooks/Hyperparameter_Tuning_and_Comparison_with_other_ML_Models.ipynb)**
This notebook contains the code for hyperparameter tuning of the models at individual station. It also shows the comparison of the selected model with other machine learning models. 


## Report Bugs
If you find inconsistencies in the code or have any question, please open the issue here or contact me at my email (ak287@uw.edu).


