#!/usr/bin/env python
# coding: utf-8

# In[639]:


import numpy as np
import IPython.display
from IPython.display import Audio
import librosa
import librosa.display
import os
from matplotlib import pyplot as plt
from math import gcd
from functools import reduce
from copy import deepcopy
from scipy.spatial import distance
import mir_eval


def spectralFlux(readFile, hopTime = 0.010, windowTime = 0.3):
    #import audio
    snd, rate = librosa.load(readFile, sr=None)
    hop = round(rate * hopTime)
    # round up to next power of 2
    wlen = int(2 ** np.ceil(np.log2(rate * windowTime)))
    # centre frames: first frame at t=0
    snd = np.concatenate([np.zeros(wlen//2), snd, np.zeros(wlen//2)])
    frameCount = int(np.floor((len(snd) - wlen) / hop + 1))
    window = np.hamming(wlen)
    prevM = np.zeros(wlen)
    
    sf = np.zeros(frameCount)
   
    for i in range(frameCount):
        start = i * hop
        frame = np.fft.fft(snd[start: start+wlen] * window)
        mag = np.abs(frame)
        sf[i] = np.mean(np.multiply(np.greater(mag, prevM), np.subtract(mag, prevM)))

    prevM = mag
    
    return sf, snd, rate

####DETECT ONSETS: Peak picking algorithm###
threshold = 2.4
hop = 0.01

#apply spectral flux
odf, snd, rate = spectralFlux("BallroomData/Jive/Albums-Latin_Jam-11.wav")

#get the time instances of each spectral flux difference
time = np.multiply(range(len(odf)), hop)
difference = np.diff(odf, axis=0)
#If the difference in spectral energy is above the threshold, it is a peak
isPeak = np.multiply(np.multiply(np.greater(odf, threshold),
                    np.greater(np.concatenate([np.zeros(1), difference]), 0)),
                    np.less(np.concatenate([difference, np.zeros((1,))]), 0))
peaks = np.nonzero(isPeak)
peakIndex = peaks[0]
peakTimes=time[peakIndex]

#plot waveform and onsets
librosa.display.waveplot(snd, rate, color='blue')

for i in peakTimes:
    plt.axvline(x=i)
    
    
#Inter onset intervals become the difference between the current frame and the previous frame
iOI= np.zeros(len(peakTimes)-1)
for i in range(len(iOI)):
    if i+1<=len(peakTimes):
        iOI[i]=peakTimes[i+1]-peakTimes[i]
#Find the most common IOI
n,bins, patches = plt.hist(iOI,100)
nmax=np.argmax(n)


plt.show()

###Beat Tracker###


#State Variables

Tempo = (1/mainCluster)*15
#first beat period is based on the tempo
beatPeriod = Tempo/120; 
agent = np.zeros(len(peakTimes))
agent_temp = np.zeros((len(peakTimes)))
alpha = 0.1
percentage = 0
#agent[0]= mainCluster

for i in range (len(peakTimes)-1):
    #increment the temporary agent by the beat period
    agent_temp[i+1] = agent[i]+(beatPeriod)
    #find closest onset to agent's current position
    closestOnset = (np.abs(peakTimes - agent_temp[i+1])).argmin()
    #if the closest onset is far away, the agent's position stays the same
    if(peakTimes[closestOnset]-agent_temp[i+1]>=0.1 or peakTimes[closestOnset]-agent_temp[i+1]<=-0.1):
        agent[i+1]=agent_temp[i+1]
    #Otherwise, agent goes to closest onset
    else:  
        agent[i+1] = peakTimes[closestOnset]
    #update beat period accordingly 
    beatPeriod=alpha*(agent[i+1]-agent[i])+(1-alpha)*beatPeriod

#play metronome with track at agent times
metronome = librosa.clicks(times=agent, sr=rate)
IPython.display.display(IPython.display.Audio(snd, rate=rate,autoplay = True))
IPython.display.display(IPython.display.Audio(metronome, rate=rate,autoplay = True))

#test data against ground truth
groundTruth = (np.loadtxt(fname ="BallroomAnnotations-master/Albums-Latin_Jam-11.beats"))
groundTruth = groundTruth[:,0]
f_measure = mir_eval.beat.f_measure(groundTruth,agent)
print(f_measure)


# In[521]:





# In[ ]:




