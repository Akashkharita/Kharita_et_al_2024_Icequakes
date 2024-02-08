from scipy import signal
import scipy
import numpy as np
import obspy
import pandas as pd
import tsfel
from scipy import signal
from scipy.fft import fft, fftfreq
import numpy as np
from scipy.signal import hilbert
from sklearn import metrics




def RSAM(data, samp_rate, datas, freq, Nm, N):
    filtered_data = obspy.signal.filter.bandpass(data, freq[0], freq[1], samp_rate)
    filtered_data = abs(filtered_data[:Nm])
    datas.append(filtered_data.reshape(-1,N).mean(axis=-1)*1.e9) # we should remove the append
    return(datas)

def DSAR(data, samp_rate, datas, freqs_names, freqs, Nm, N):
    # compute dsar
    data = scipy.integrate.cumtrapz(data, dx=1./100, initial=0) # vel to disp
    data -= np.mean(data) # detrend('mean')
    j = freqs_names.index('mf')
    mfd = obspy.signal.filter.bandpass(data, freqs[j][0], freqs[j][1], samp_rate)
    mfd = abs(mfd[:Nm])
    mfd = mfd.reshape(-1,N).mean(axis=-1)
    j = freqs_names.index('hf')
    hfd = obspy.signal.filter.bandpass(data, freqs[j][0], freqs[j][1], samp_rate)
    hfd = abs(hfd[:Nm])
    hfd = hfd.reshape(-1,N).mean(axis=-1)
    dsar = mfd/hfd
    datas.append(dsar)
    return(datas, dsar)

def nDSAR(dsar):
    return dsar/scipy.stats.zscore(dsar)

def compute_envelope(data):

    
    analytic_signal = hilbert(data)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope  
    
def compute_physical_features(tr, envfilter = True, freq_bands = [[0.1,1],[1,3],[3,10],[10,20],[20,50]], env_filt = [0.01]):
    
    ''' This functions computes features used by Hibert's research group.
    The list of features can be found at - https://www.sciencedirect.com/science/article/pii/S0377027316303948
    and consult this for a full meaning of individual feature. - https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016GL070709
    
    
    tr = Trace of the raw seismogram, it could be a filtered seismogram as observed in previous studies. 
    
    Look at this code for reference - https://github.com/krischer/mess_2019/tree/master/3_wednesday
    
    '''
    
    
    

    
 

  
    
    env = compute_envelope(tr.data)
    
    if envfilter == True:
        sos = signal.butter(1, env_filt, 'lp', fs = tr.stats.sampling_rate,  output = 'sos')
        env  = signal.sosfilt(sos, env)
    
    
    NATT = 62
    all_attr  = np.empty((1,NATT), dtype = float)
    
    f11 = freq_bands[0][0]
    f12 = freq_bands[0][1]
    f21 = freq_bands[1][0]
    f22 = freq_bands[1][1]
    f31 = freq_bands[2][0]
    f32 = freq_bands[2][1]
    f41 = freq_bands[3][0]
    f42 = freq_bands[3][1]
    f51 = freq_bands[4][0]
    f52 = freq_bands[4][1]
    
    attributes = ['Window_Length', 'RappMaxMean', 'RappMaxMedian', 'AsDec','KurtoSig', 'KurtoEnv', 'SkewSig', 
                  'SkewEnv', 'CorPeakNumber', 'Energy1/3Cor', 'Energy2/3Cor', 'int_ratio','E_'+str(f11)+'_'+str(f12),'E_'+str(f21)+'_'+str(f22),
                  'E_'+str(f31)+'_'+str(f32), 'E_'+str(f41)+'_'+str(f42),
                  'E_'+str(f51)+'_'+str(f52),
                  'Kurt_'+str(f11)+'_'+str(f12),
                  'Kurt_'+str(f21)+'_'+str(f22),
                  'Kurt_'+str(f31)+'_'+str(f32),
                  'Kurt_'+str(f41)+'_'+str(f42),
                  'Kurt_'+str(f51)+'_'+str(f52),
                  'RMSDecPhaseLine', 'MeanFFT','MaxFFT','FMaxFFT','MedianFFT','VarFFT','FCentroid','Fquart1','Fquart3','NPeakFFT',
                  'MeanPeaksFFT','E1FFT','E2FFT','E3FFT','E4FFT', 'Gamma1', 'Gamma2', 'Gamma','KurtoMaxDFT','KurtoMedianDFT','MaxOverMeanDFT',
                  'MaxOverMedianDFT','NbrPeaksMaxDFT','NbrPeaksMeanDFT','NbrPeaksMedianDFT','45/46','45/47','NbrPeaksCentralFreq','NbrPeaksMaxFreq',
                  '50/51', 'DistMaxMeanFreqDTF','DistMaxMedianFreqDTF','DistQ2Q1DFT','DistQ3Q2DFT','DistQ3Q1DFT',
                  'Peak_Envelope_Amplitude', 'Average_Envelope_Amplitude','Envelope_Area', 'Envelope_Velocity', 'Envelope_Rise_Time']
    
    
    auto = np.correlate(tr.data, tr.data, 'same') ## autocorrelation function
    
    t = tr.times()
    
    # family number 1: Based on waveforms. 
    all_attr[0,0] = t[-1] - t[0]   # Window Length
    all_attr[0,1] = np.nanmax(env)/np.nanmean(env)   # Ratio of max and mean of envelope
    all_attr[0,2] = np.nanmax(env)/np.nanmedian(env) # Ratio of max and median of envelope.
    all_attr[0,3] = (t[np.argmax(env)] - t[0])/(t[-1]-t[np.argmax(env)]) # Ratio of ascending and descending times of envelope
    all_attr[0,4] = scipy.stats.kurtosis(tr.data) # Kurtosis of signal
    all_attr[0,5] = scipy.stats.kurtosis(env) # Kurtosis of envelope
    all_attr[0,6] = scipy.stats.skew(tr.data) # Skewness of signal
    all_attr[0,7] = scipy.stats.skew(env)  # Skewness of envelope
    all_attr[0,8] = len(scipy.signal.find_peaks(auto)[0]) # Computing the number of peaks 
    all_attr[0,9] = np.trapz(y = auto[0:int(len(auto)/3)]) # computing energy in 1/3 of autocorr
    all_attr[0,10] = np.trapz(y = auto[int(len(auto)/3):len(auto)]) #computing energy in last 2/3 of autocorr
    all_attr[0,11] = np.trapz(y = auto[0:int(len(auto)/3)])/np.trapz(y = auto[int(len(auto)/3):len(auto)]) #int_ratio
    
    
    
    
    
    
    
    # family number 2: Spectral waveforms
    ## Defining the filters
    # We are using a butterworth filter two corners 
    d = tr.data 
    try:
        sos_0p1_1 = signal.butter(N = 2, Wn= [freq_bands[0][0],freq_bands[0][1]], btype = 'bp', fs=tr.stats.sampling_rate, output='sos')
        # filter = 0.1 - 1 Hz
        filtered_0p1_1 = signal.sosfilt(sos_0p_1, d)
        # filtering the waveforms
        env_0p1_1 = compute_envelope(filtered_0p1_1)
        all_attr[0,12] = np.log10(np.trapz(y = abs(env_0p1_1), x = tr.times())) # Area of the filtered envelope
        all_attr[0,17] = scipy.stats.kurtosis(filtered_0p1_1) # Kurtosis of filtered signal 
    except:
        pass
    
    try:
        
         ## Filter between  1 - 3 Hz
        sos_1_3 = signal.butter(N = 2, Wn= [freq_bands[1][0],freq_bands[1][1]], btype = 'bp', fs=tr.stats.sampling_rate, output='sos')
        # filter = 1 - 3 Hz 
        filtered_1_3 = signal.sosfilt(sos_1_3, d)
        # filtering the waveforms
        env_1_3 = compute_envelope(filtered_1_3)
        all_attr[0,13] = np.log10(np.trapz(y = abs(env_1_3), x = tr.times())) # Area of the filtered envelope
        all_attr[0,18] = scipy.stats.kurtosis(filtered_1_3) # Kurtosis of filtered signal
    except:
        pass
    
    try:
        
        ## Filter between 3 - 10 Hz
        sos_3_10 =  signal.butter(N = 2, Wn= [freq_bands[2][0],freq_bands[2][1]], btype = 'bp', fs=tr.stats.sampling_rate, output='sos')
        filtered_3_10 = signal.sosfilt(sos_3_10, d)
        env_3_10 = compute_envelope(filtered_3_10)
        all_attr[0,14] = np.log10(np.trapz(y = abs(env_3_10), x = tr.times()))
        all_attr[0,19] = scipy.stats.kurtosis(filtered_3_10)
    except:
        pass
    
        
        
    try:
        
        ## Filter between 10 - 20 Hz. 
        sos_10_20 = signal.butter(N = 2, Wn= [freq_bands[3][0],freq_bands[3][1]], btype = 'bp', fs=tr.stats.sampling_rate, output='sos')
        filtered_10_20 = signal.sosfilt(sos_10_20, d)
        env_10_20 = compute_envelope(filtered_10_20)
        all_attr[0,15] = np.log10(np.trapz(y = abs(env_10_20), x = tr.times()))
        all_attr[0,20] = scipy.stats.kurtosis(filtered_10_20)
    except:
        pass
    
        
    try:
        
        ## Filter between 20. - 50 Hz
        sos_20_50 = signal.butter(N = 2, Wn= [freq_bands[4][0],freq_bands[4][1]], btype = 'bp', fs=tr.stats.sampling_rate, output='sos')
        filtered_20_50 = signal.sosfilt(sos_20_50, d)
        env_20_50 = compute_envelope(filtered_20_50)
        all_attr[0,16] = np.log10(np.trapz(y = abs(env_20_50), x = tr.times()))
        all_attr[0,21] = scipy.stats.kurtosis(filtered_20_50)
        
    except:
        pass
    
    
    l = np.nanmax(env) - ((np.nanmax(env)/(t[-1]-t[np.nanargmax(env)]))*(t))
    all_attr[0,22]  =  np.sqrt(np.nanmean((env - l)**(2)))  #RMSDecPhaseLine
    
    
    
    
    
    ## Spectral attributes
    
    
    ft  = abs(fft(tr.data))   ## Discrete Fast Fourtier Transform
    freq = fftfreq(len(tr.data), d= tr.stats.delta)  ## Obtaining the frequencies. 
    ft = ft[0:len(ft)//2]  ## Because it is symmeteric across zero. 
    freq = freq[0:len(freq)//2]
    

    all_attr[0,23] = np.nanmean(ft)  ## Mean FFT
    all_attr[0,24] = np.nanmax(ft)   ## MaxFFT
    all_attr[0,25] = freq[np.nanargmax(ft)] #FMaxFFT : Dominant frequency. 
    all_attr[0,26] = np.nanmedian(ft) #MedianFFT
    all_attr[0,27] = np.nanvar(ft)   #VarFFT
    all_attr[0,28] = np.dot(freq,ft)/(np.sum(ft))  # centroid frequency: To get an idea of frequency content of the signal. If the signal is dominated 
    # by higher frequencies, its centroid frequency will be higher. 
    
    
    all_attr[0,29] = np.dot(freq[0:len(ft)//4],ft[0:len(ft)//4])/(np.sum(ft[0:len(ft)//4])) 
    # centroid frequency of first quartile
    
    
    all_attr[0,30] = np.dot(freq[len(ft)//2:int(0.75*len(ft))],ft[len(ft)//2:int(0.75*len(ft))])/(np.sum(ft[len(ft)//2:int(0.75*len(ft))])) 
    # centroid frequency of third quartile 
    
    all_attr[0,31] = len(signal.find_peaks(ft, height = 0.75*np.nanmax(ft))[0])
    # NpeakFFT. (Number of peaks that have atleast 75%)
    
    
    
    
    all_attr[0,32] = np.nanmean(ft[signal.find_peaks(ft, height = 0)[0]])
    # MeanPeaksFFT =. mean of peaks in the frequency spectrum
    
    
    # Energy of fft curve in four quadrants. 
    all_attr[0,33] = np.trapz(y = ft[0:len(ft)//4],x = freq[0:len(ft)//4])
    all_attr[0,34] = np.trapz(y = ft[len(ft)//4:len(ft)//2],x = freq[len(ft)//4:len(ft)//2])
    all_attr[0,35] = np.trapz(y = ft[len(ft)//2:int(3*len(ft)//4)],x = freq[len(ft)//2:int(3*len(ft)//4)])
    all_attr[0,36] = np.trapz(y = ft[int(3*len(ft)//4):len(ft)],x = freq[int(3*len(ft)//4):len(ft)])
    
    
    # Gamma 1 : Spectral Centroid. - Measure of signals spectral characteristics. 
    all_attr[0,37] = np.dot(freq, ft**(2))/np.sum(ft**(2))
    
    # Gamma 2 : Gyration radius - it refers to the distribution of mass around the centre of mass, 
    # Suppose any arbitrary shaped body is converted into a circle around its centre of mass, its the radius 
    # of that circle. 
    all_attr[0,38] = (np.dot(freq**(2), ft**(2))/np.sum(ft**(2)))**(0.5)
    
    # Gamma 1 and 2: This will be a measure of spectral centroid width
    all_attr[0,39] = (all_attr[0,37]**(2) - all_attr[0,38]**(2))**(0.5)
    
    
    
    ## Spectrogram attributes
    
    f, t, Sxx = signal.spectrogram(tr.data, fs = tr.stats.sampling_rate)
    all_attr[0,40] = scipy.stats.kurtosis(np.nanmax(abs(Sxx), axis=0)) 
    ## KurtoMaxDFT  - tracking the maximum along spectrogram and computing
    # its flatness
    
    all_attr[0,41] = scipy.stats.kurtosis(np.nanmedian(abs(Sxx), axis=0))
    ## KurtoMedianDFT -  tracking the median across the spectrogram 
    
    all_attr[0,42] = np.nanmean(np.nanmax(abs(Sxx), axis=0)/np.nanmean(abs(Sxx), axis=0))
    # MaxOverMeanDFT - mean of the series that is obtained by dividing the max over mean of the spectrogram. 
    
    
    all_attr[0,43] = np.nanmean(np.nanmax(abs(Sxx), axis=0)/np.nanmedian(abs(Sxx), axis=0))
    # MaxOverMedianDFT - mean of the series that is obtained by dividing the max over median of the spectrogram. 
    
    
    all_attr[0,44] = len(signal.find_peaks(np.nanmax(abs(Sxx), axis=0))[0])
    # NbrPeaksMaxDFT. - number of peaks in the maximum DFT. 
    
    all_attr[0,45] = len(signal.find_peaks(np.nanmean(abs(Sxx), axis=0))[0])
    # NbrPeaksMeanDFT - number of peaks in the mean DFT
    
    
    all_attr[0,46] = len(signal.find_peaks(np.nanmedian(abs(Sxx), axis=0))[0])
    # NbrPeaksMedianDFT  - number of peaks in the median DFT
    
    all_attr[0,47] = all_attr[0,44]/all_attr[0,45]
    # Roughness ratio. 
    
    all_attr[0,48] = all_attr[0,44]/all_attr[0,46]
    # Another roughness ratio
    
    all_attr[0,49] = len(signal.find_peaks(np.dot(f,abs(Sxx))/np.sum(abs(Sxx), axis=0))[0])
    # Number of peaks in the series obtained by taking central frequency in each time bin
    # (temporal evolution of central frequency)
    # So now this function is np.dot(f, abs(Sxx)) - taking the dot product and dividing by its sum would compute its central frequency
    
    
    all_attr[0,50] = len(signal.find_peaks(np.nanmax(abs(Sxx), axis=0))[0])
    # Number of peaks in the temporal evolution of maximum frequency. 
    
    
    
    all_attr[0,51] = all_attr[0,49]/all_attr[0,50]
    # roughness ratio. 
    
    all_attr[0,52] = np.nanmean(np.nanmax(abs(Sxx), axis=0) - np.nanmean(abs(Sxx), axis=0))
    # Mean distance between the curves of temporal evolution of the maximum and the mean frequency in 
    # each time bin. 
    
    
    all_attr[0,53] = np.nanmean(np.nanmax(abs(Sxx), axis=0) - np.nanmedian(abs(Sxx), axis=0))
    # Mean distance between the curves of temporal evolution of the maximum and median frequency. 
    
    
     
    Sq1  = abs(Sxx[:,0:len(t)//4])
    Sq2  = abs(Sxx[:,len(t)//4:2*int(len(t)//4)])
    Sq3  = abs(Sxx[:,2*int(len(t)//4):3*int(len(t)//4)])
    Sq4  = abs(Sxx[:,3*int(len(t)//4):4*int(len(t)//4)])
                         
      
                         
                     
    all_attr[0,54] = np.nanmean(np.dot(f,Sq2)/np.sum(Sq2, axis=0) - np.dot(f,Sq1)/np.sum(Sq1, axis=0))
    # DistQ2Q1DFT - Mean distance between the median of the second and first quartile
    
    all_attr[0,55] = np.nanmean(np.dot(f,Sq3)/np.sum(Sq3, axis=0) - np.dot(f,Sq2)/np.sum(Sq2, axis=0))
    # DistQ23Q2DFT - Mean distance between the median of the second and first quartile
    
    all_attr[0,56] = np.nanmean(np.dot(f,Sq3)/np.sum(Sq3, axis=0) - np.dot(f,Sq1)/np.sum(Sq1, axis=0))
    # DistQ3Q1DFT - Mean distance between the median of the second and first quartile

   

    t = tr.times()
    
    all_attr[0,57] = np.nanmax(env) 
    # Peak envelope amplitude
    
    all_attr[0,58] = np.nanmean(env)
    # Mean envelope amplitude
    
    all_attr[0,59] = metrics.auc(t,env)
    # Area under the curve of envelope
    
    all_attr[0,60] = (metrics.auc(t,env))/(t[-1] - t[0])
    # Area/Duration  = Velocity 
    
    all_attr[0,61] = t[np.argmax(env)] - t[0]
    # Risetime. 
    
    feature = pd.DataFrame(data = all_attr, columns = attributes)
    return feature 





def compute_features(slide_id, df_good, feature_type = 'physics based', envfilter = False, duration = 'on', window = 240):
    
    
    
    
    
    
    Features = pd.DataFrame([])

    for i in tqdm(range(len(slide_id))):

        
            df_temp = df_good.iloc[np.where(df_good['eventid'] == slide_id[i])[0]]

            data_files = glob('../Data/Exotic_IRIS_4minutes/waveforms/'+str(slide_id[i])+'/*')
            
            if len(data_files) != 0:

                
                # we are extracting variables from the dataframe
                snrs = df_temp['snr'].values
                stns = df_temp['station'].values
                types = df_temp['type'].values
                vols = df_temp['volume'].values
                dists = df_temp['distance'].values
                sources = df_temp['subtype'].values

                # obtaining the stored waveforms
                st = obspy.read('../Data/Exotic_IRIS_4minutes/waveforms/'+str(slide_id[i])+'/*')

                # obtaining the vertical component
                st_z = st.select(channel = '*HZ')
                
                # obtaining the instrument response inventory
                inv = obspy.read_inventory('../Data/Exotic_IRIS_4minutes/stations/'+str(slide_id[i])+'/*')

                #print(inv)
                
                #removing the response 
                #st_z.remove_response(inv)
                

                # detrending
                #st_z.detrend()

                # filtering
                #st_z.filter('bandpass', freqmin = 0.5, freqmax= 10)




                ## setting the order of the waveforms
                order = np.argsort(dists)


                #fig, ax = plt.subplots(nrows = len(st_z), ncols = 2,figsize = [15,2.0*len(st_z)], sharex = False, gridspec_kw={'width_ratios': [1, 1]})
                #fig.suptitle('EventID: '+str(slide_id[i])+' Volume:'+str(vols[0])+' '+sources[0], fontweight = 'bold', y=0.999)


                for j in range(len(order)):

                        try:
                            
                                    tr = st_z.select(station = stns[order[j]])[0]
                                    inv_stn = inv.select(station = stns[order[j]])
                                    
                                    starttime = tr.stats.starttime
                                    
                                    tr.remove_response(inv_stn)
                
                                    
                                    #tr.trim(starttime, starttime + 239.95)           
                                    
                                    tr.detrend()
                                    tr.taper(0.01)
                                    tr.filter('bandpass', freqmin = 0.5, freqmax = 10)

                                    env = obspy.signal.filter.envelope(tr.data)
                                    sos = signal.butter(1, 0.02, 'lp', fs = tr.stats.sampling_rate,  output = 'sos')
                                    env_filt  = signal.sosfilt(sos, env)


                                    sr = tr.stats.sampling_rate 


                                    # Defining the duration by 5-95% energy method. 
                                    # For this we define the starttime as coming at 50s, 10s before the defined starttimes in the IRIS catalog. 

                                    if duration == 'on':
                                        ## This is the code for computing 5-95% duration but since we are concerned about initial 
                                        ## four minutes of the data. Then it should not be a problem. 
                                        x = tr.times()[int(50*sr):-1]
                                        y = env_filt[int(50*sr):-1]

                                        # integrating the envelope to define cumulative area
                                        y_int = sc.integrate.cumtrapz(y, x, initial = 0)



                                        ####
                                        te = x[np.where(y_int > np.percentile(y_int,95))[0][0]]
                                        ts = x[np.where(y_int > np.percentile(y_int,5))[0][0]]

                                        starttime = tr.stats.starttime
                                        tr  = tr.trim(starttime+ts, starttime+te)
                                        env = obspy.signal.filter.envelope(tr.data)
                                        sos = signal.butter(1, 0.01, 'lp', fs = tr.stats.sampling_rate,  output = 'sos')
                                        env_filt  = signal.sosfilt(sos, env)
                                    
                                    else:
                                        tr.trim(starttime, starttime + window)

                                 
                  

                                    if feature_type == 'physics based':
                                        df = compute_physical_features(tr, envfilter = envfilter)

                                    if feature_type == 'tsfel':
                                        cfg = tsfel.get_features_by_domain()
                                        df = tsfel.time_series_features_extractor(cfg, tr.data, fs= sr, window_size=len(tr.data)) 
                                        


                                    df['Event_ID'] = slide_id[i]
                                    df['Volume'] = vols[0]
                                    df['Event_Type'] = sources[order[j]]
                                    df['Station'] = stns[order[j]]
                                    df['Distance'] = dists[order[j]]
                                    df['Startttime'] = tr.stats.starttime
                                    df['Endtime'] = tr.stats.endtime
                                    df['SNR'] = snrs[order[j]]


                                    Features = pd.concat([Features,df])
   
                        except:
                            pass
        


                
    return Features
