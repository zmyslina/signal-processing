# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 14:44:17 2022

@author: kopex
"""

from __future__ import print_function 
import numpy as np 
import matplotlib.pyplot as plt 
import librosa 
import librosa.display 
import soundfile as sf 
  
y, sr = librosa.load('piosenka.wav', duration=20) 
  
S_full, phase = librosa.magphase(librosa.stft(y)) 
  
idx = slice(*librosa.time_to_frames([1, 20], sr=sr)) 
  
S_filter = librosa.decompose.nn_filter(S_full,aggregate=np.median,metric='cosine',width=int(librosa.time_to_frames(2, sr=sr))) 
  
S_filter = np.minimum(S_full, S_filter) 
  
margin_i, margin_v = 5, 20 
power = 5 
mask_m = librosa.util.softmask(S_filter,margin_i * (S_full - S_filter),power=power) 
mask_g = librosa.util.softmask(S_full - S_filter,margin_v * S_filter,power=power) 
  
glos = mask_g * S_full 
muzyka = mask_m * S_full 
  
plt.figure(figsize=(12, 8)) 
plt.subplot(3, 1, 1) 
librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),y_axis='log', sr=sr) 
plt.title('Cała piosenka') 
plt.colorbar() 
plt.subplot(3, 1, 2) 
librosa.display.specshow(librosa.amplitude_to_db(muzyka[:, idx], ref=np.max),y_axis='log', sr=sr) 
plt.title('Muzyka') 
plt.colorbar() 
plt.subplot(3, 1, 3) 
librosa.display.specshow(librosa.amplitude_to_db(glos[:, idx], ref=np.max),y_axis='log', x_axis='time', sr=sr) 
plt.title('Głos') 
plt.colorbar() 
plt.tight_layout() 
plt.show() 
  
  
y_glos = librosa.istft(glos) 
sf.write("glos.wav", y_glos, samplerate=sr, subtype='PCM_24') 
  
y_muzyka = librosa.istft(muzyka) 
sf.write("muzyka.wav", y_muzyka, samplerate=sr, subtype='PCM_24') 
  
calosc = muzyka + glos  
y_calosc = librosa.istft(calosc) 
sf.write("calosc.wav", y_calosc, samplerate=sr, subtype='PCM_24') 
