# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 09:45:43 2022

@author: Aayush
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 
import ampd
import scipy 
from scipy import signal, stats
from scipy.signal import find_peaks_cwt
from ampd import find_peaks, find_peaks_original, find_peaks_adaptive
import hampel as ha 

plt.close('all')

''' function definitions required'''

def merge_files(loc, lower, upper):
    
    for counter,i in enumerate(sorted(os.listdir(loc))):
        if counter > upper or counter < lower:
            continue 
        path = os.path.join(loc, 'right_ankle_5(MA)_{}.csv'.format(counter +1))
        print('adding file number {}'.format(counter))
        if counter == lower:
            df = pd.read_csv(path)
        else:
            df = df.append(pd.read_csv(path))
    return df 

def get_rmssd(variable):
    new_list = []
    for i in range(len(variable)):
        item = variable[i]**2
        new_list.append(item)
    new_array = np.asarray(new_list)
    dif_rms = np.sqrt(np.sum(new_array)/len(variable))
    return dif_rms

def get_std(variable):
    return np.std(variable)

def moving_average(sig, window_size):
    i = 0
    moving_averages = []
    while i < len(sig) - window_size + 1:
        window = sig[i : i + window_size]
        window_average = round(sum(window) / window_size, 2) 
        moving_averages.append(window_average)
        i += 1
    return np.asarray(moving_averages)

def filter_signal(signals,sampling_fs, f_order, low_cut, high_cut, window_size):
    f_order = f_order
    nyq = 0.5 * sampling_fs
    fc_2 = high_cut/nyq
    fc_1 =  low_cut/nyq
    b_1,a_1 = signal.butter(f_order,[fc_1,fc_2], btype='bandpass') 
    ppg_2 = signal.filtfilt(b_1,a_1, signals)
    y_1 = ha.hampel(pd.Series(ppg_2), window_size = window_size, n = 3,imputation = True)
    #y_2 = signal.medfilt(y_1, kernel_size = kernel_size)
    
    return y_1

def get_hrv(signal, sampling_fs):
    times = np.arange(0,len(signal))/sampling_fs
    sig = find_peaks_original(signal)
    tp = []
    for i in times[sig]:
        tp.append(i)
    
    difference = [a - tp[i - 1] for i,a in enumerate(tp)][1:]
    tt = np.arange(0,len(difference))/sampling_fs
    index_point = np.where(tt == 60)
    # cut_hrv = difference[:index_point]
    # dif_rms = get_rmssd(cut_hrv)
    # std_1 = get_std(cut_hrv)
    
    print(type(index_point))


def heart_rate(signals, sampling_fs):
    number_of_points = []
    tim = np.arange(0,len(signals))/sampling_fs
    Y_2 = np.fft.rfft(signals)
    f_3 = np.arange(0,len(Y_2))*sampling_fs/len(signals)
    max_freq = np.where(Y_2 == max(Y_2))
    
    signal_peaks = find_peaks_original(signals)
    
    for i in range(len(signal_peaks)):
        if signal_peaks[i] >= np.where(tim == 10):
            break
        else:
            number_of_points.append(signal_peaks[i])
            
    heart_rate_t = len(number_of_points)/10
    heart_rate_t = heart_rate_t * 60
    heart_rate_f = f_3[max_freq] * 60
    heart_rates = np.asarray([heart_rate_f, heart_rate_t])
    mean_rate = np.mean(heart_rates)
    
    return mean_rate
    
# apply power spectrum rather than fast fourier transform

'''loc = 'D:\Milbotix internship\HR data\save 10'
dataframe_1 = merge_files(loc,0,2)
dataframe_1.rename(columns = {'Channel A':'Volts (V)'},
              inplace = True)
signal_1 = dataframe_1['Volts (V)']
signal_1_1 = signal_1 - np.mean(signal_1)
signal_1_1 = signal_1_1.dropna()
t_1 = np.arange(0,len(signal_1_1))/150


plt.figure(figsize = (20,8))
plt.title('signal with motion artefact', fontsize = 25)
plt.plot(t_1,signal_1_1)
plt.xlabel('Time (s)', fontsize = 20)
plt.ylabel('Voltage (V)', fontsize = 20)



f_order = 4
nyq = 0.5 * 150
fc_2 = 2.3/nyq
fc_1 =  0.7/nyq
b_1,a_1 = signal.butter(f_order,[fc_1,fc_2], btype='bandpass') 
ppg_2 = signal.filtfilt(b_1,a_1, signal_1_1)
y_1 = ha.hampel(pd.Series(ppg_2), window_size = 7, n = 3,imputation = True)
y_2 = signal.medfilt(y_1, kernel_size = 15)


# keep the window size tp be either 3,5,7
# kepp kernel size to be about 3,5,7,9
# keep standard deviations 2 or 3
  
t_5 = np.arange(0,len(y_1))/150
plt.figure(figsize = (20,8))
plt.title('filtered signal', fontsize = 25)
plt.plot(t_5,y_2, label = 'Hampel filter output')
# plt.plot(t_5,ppg_2, label = 'signal before hampel' )
plt.xlabel('Time (s)', fontsize = 20)
plt.ylabel('Voltage (V)', fontsize = 20)
plt.legend()

Y_2 = np.fft.rfft(y_1)
f_3 = np.arange(0,len(Y_2))*150/len(y_1)
max_out = np.where(Y_2 == max(Y_2))

plt.figure(figsize = (20,8))
plt.title('Filtered signal frequency response', fontsize = 25)
plt.plot(f_3,np.abs(Y_2))
plt.plot(f_3[max_out],Y_2[max_out], 'bo', label = 'max peak')
plt.xlabel('Frequency (Hz)', fontsize = 20)
plt.ylabel('Amplitude', fontsize = 20)



# # find the peaks in the signal using ampd

sig = find_peaks_original(y_2)

plt.figure(figsize = (20,8))
plt.title('Find the peaks in filtered signal', fontsize = 25)
plt.plot(t_5[sig], y_2[sig], 'bo', label = 'Peaks')
plt.plot(t_5,y_2, label = 'filtered signal')
plt.xlabel('Time (s)', fontsize = 20)
print(np.where(t_5 == 10))

number_of_points = []
for i in range(len(sig)):
    if sig[i] >= np.where(t_5 == 10):
        break
    else:
        number_of_points.append(sig[i])
heart_rate_t = len(number_of_points)/10
heart_rate_t = heart_rate_t * 60
heart_rate_f = f_3[max_out] * 60
print('the heart rate from time domain is {}'.format(heart_rate_t))
print('the heart rate from frequency domain is {}'.format(heart_rate_f))

# get time difference between rach interval

plt.figure(figsize = (20,8))
plt.title('Normal heart rate variability', fontsize = 25)
plt.plot(tt,difference)
plt.xlabel('Time (s)')
plt.ylabel('difference')

dif_rms = get_rms(difference)
std_1 = get_std(difference)
heart_rates = np.asarray([heart_rate_f, heart_rate_t])
print('the RMSSD is {}'.format(heart_rate_t))
print('the heart rate from frequency domain is {}'.format(heart_rate_t))
print('the average heart rate is {}'.format(np.mean(heart_rates)))'''
