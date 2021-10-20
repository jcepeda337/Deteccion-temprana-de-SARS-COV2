"""
DIAGNÓSTICO TEMPRANO DE COVID 
05/2021 VERSION 1.0

DATASET: https://zenodo.org/record/4048312#.YLktNMzPxEb
"""

import numpy as np
import pandas as pd
import os
import sklearn
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sys
import csv
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.decomposition import PCA
from sklearn.metrics import matthews_corrcoef
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from scipy import signal
from sklearn.model_selection import LeavePOut
from scipy.io import wavfile
from scipy.signal import butter,filtfilt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from scipy.stats import kurtosis
import scipy.signal as signal
from google.colab import files
from scipy.integrate import simps
import matplotlib.patches as mpatches
from imblearn.combine import SMOTEENN
from scipy.signal import butter,filtfilt
from scipy.signal import cwt
from scipy.signal import hilbert
from scipy.signal import resample
from scipy.signal import decimate
from imblearn.over_sampling import SMOTE, ADASYN
from scipy.signal import spectrogram
from sklearn.metrics import confusion_matrix
from scipy.signal.windows import get_window
from sklearn.preprocessing import StandardScaler

!apt install ffmpeg
from google.colab import drive
drive.mount('/content/drive')

!pip install unrar
!unzip /content/drive/MyDrive/Datasetzip/output.zip

#Proceso para convertir los archivos a .wav
df = pd.read_csv('/content/public_dataset/metadata_compiled.csv')
names_to_convert = df.uuid.to_numpy()
samples_to_extract_covid = df.status.to_numpy()
samples_to_extract_probability = df.cough_detected.to_numpy()
path = '/content/public_dataset/'
folder_covid = '/content/Status/Covid-19/'
folder_healthy = '/content/Status/Healthy/'
for counter, name in enumerate(names_to_convert):
        if os.path.isfile(path + name + '.webm'):
            print(name,samples_to_extract_covid[counter],samples_to_extract_probability[counter])
            if (samples_to_extract_covid[counter] == 'healthy' and samples_to_extract_probability[counter] >= 0.11):
              subprocess.call(["ffmpeg", "-i", path+name+".webm", folder_healthy+name+".wav"])
            if (samples_to_extract_covid[counter] == 'COVID-19' and samples_to_extract_probability[counter] >= 0.11):
              subprocess.call(["ffmpeg", "-i", path+name+".webm", folder_covid+name+".wav"])
        elif os.path.isfile(path + name + '.ogg'):
          print(name,samples_to_extract_covid[counter],samples_to_extract_probability[counter])
          if (samples_to_extract_covid[counter] == 'healthy' and samples_to_extract_probability[counter] >= 0.11):
              subprocess.call(["ffmpeg", "-i", path+name+".ogg", folder_healthy+name+".wav"])
          if (samples_to_extract_covid[counter] == 'COVID-19' and samples_to_extract_probability[counter] >= 0.11):
              subprocess.call(["ffmpeg", "-i", path+name+".ogg", folder_covid+name+".wav"])

#segmentar:
def cut_signal(x,fs):
    rms = np.sqrt(np.mean(np.square(x)))
    umbral_low = 0.1 * rms
    umbral_high =  2*rms
    senal_recortada = []
    padding = round(fs*0.2) # pasa a numero entero la muestra que empieza 200ms antes de la muestra que sobrepasa el umbral
    cough_start = 0
    cough_mask = np.array([False]*len(x))
    count_aux = 0
    cough_in_progress = False
    activacion_aux = True

    for counter, muestra in enumerate(x**2):
      if cough_in_progress:
        if count_aux < (cough_start + 48000):
          count_aux = count_aux + 1
          senal_recortada.append(x[count_aux])
          cough_mask[cough_start:count_aux] = True
          activacion_aux = False
      
      else:
        if muestra > umbral_high and activacion_aux == True:
          cough_in_progress = True
          if (counter-padding>=0):
            cough_start = counter-padding
            count_aux = cough_start
          else:
            cough_start = 0
    salida = np.array(senal_recortada)
    return salida

# Preprocesamiento:

# 1. Convertir el audio a mono si es estereo
def preprocesamiento (tos):
  if len(tos.shape)>1:
    tos = np.mean (tos, axis = 1)
  # 2. Normalizar la señal
  tos =  tos/(np.max(np.abs(tos))) 

  # 3. Aplicar un filtro de 4 orden 
  fs = 48000
  frecuencia_corte = 10000
  frecuencia_diezmado = frecuencia_corte * 2
  nyq = 0.5*fs
  Wn = frecuencia_corte / nyq
  # Funcion que retorna los coeficientes del filtro:
  b, a = signal.butter(4, Wn, 'low', analog=False)
  senal_filtrada = filtfilt (b,a,tos)
  # 4. Reducir la resolucion de la señal despues de aplicar el filtro:
  # Factor de diezmado M = input rate / output rate
  M = int (fs / frecuencia_diezmado) 
  tos = signal.decimate (tos, M )
  senal_salida = np.float32(tos)
  return np.float32(senal_salida), frecuencia_diezmado

# Extraccion de caracteristicas:

# Espectrograma de mel :
def MFCC(data):
  fs, cough = data
  n_mfcc = 13
  mfcc = librosa.feature.mfcc(y = cough, sr = fs, n_mfcc = n_mfcc)
  mfcc_mean = mfcc.mean(axis=1)
  print(mfcc_mean)
  mfcc_std = mfcc.std(axis=1)
  print(mfcc_std)
  mfcc = np.append(mfcc_mean,mfcc_std)
  print(mfcc)
  return mfcc

# Zero crossing rate:
def ZCR (data):
  fs, cough = data
  zcr = librosa.feature.zero_crossing_rate(cough)
  return np.mean(zcr)

# Kurtosis
def kurtosis (data):
  fs,cough = data 
  magnitudes = np.abs(np.fft.rfft(cough)) 
  length = len(cough)
  sum_mag = np.sum(magnitudes)
  freqs = np.abs(np.fft.fftfreq(length, 1.0/fs)[:length//2+1]) 
  spectral_centroid = np.sum(magnitudes*freqs) / np.sum(magnitudes)
  spec_spread = np.sqrt(np.sum(((freqs-spectral_centroid)**2)*magnitudes) / sum_mag)
  spec_kurtosis =  np.sum(((freqs-spectral_centroid)**4)*magnitudes) / ((spec_spread**4)*sum_mag)
  return spec_kurtosis

# Centroide espectral
def spectral_cen (data):
  fs,cough = data
  spec_cent = librosa.feature.spectral_centroid(cough, fs)
  return np.mean(spec_cent)


# Roll off spectral
def roll_off (data):
  fs,cough = data
  rolloff = librosa.feature.spectral_rolloff(cough, fs)
  return np.mean(rolloff)

# Desviacion estandar
def std (data):
  fs, cough = data
  name = ['standard_deviation']
  standar_deviation = np.std(cough)
  return standar_deviation

# Factor de cresta
def factor_cresta (data):
  fs, cough = data
  l = len(cough)
  x_peak = np.amax(np.absolute(cough))
  x_rms = np.sqrt(np.mean(np.square(cough)))
  creast_factor = x_peak/x_rms
  return creast_factor

# RMS Power
def RMS(data):
  fs, cough = data
  name = ['RMS']
  rms = np.sqrt(np.mean(np.square(cough)))
  return rms

# Ancho de banda espectral
def spectral_bandwidth(data):
  fs,cough = data
  spec_bw = librosa.feature.spectral_bandwidth(cough,fs)
  return np.mean(spec_bw)

# Densidad espectral de potencia
def PSD (data):
  FREQ_CUTS = [(0,200),(300,425),(500,650),(950,1150),(1400,1800),(2300,2400),(2850,2950),(3800,3900)]
  fs, cough = data
  feat = []
  nperseg = min(900,len(cough))
  noverlap=min(600,int(nperseg/2))
  freqs, psd = signal.welch(cough, fs, nperseg=nperseg, noverlap=noverlap)
  dx_freq = freqs[1]-freqs[0]
  total_power = simps(psd, dx=dx_freq)
  for lf, hf in FREQ_CUTS:
      idx_band = np.logical_and(freqs >= lf, freqs <= hf)
      band_power = simps(psd[idx_band], dx=dx_freq)
      feat.append(band_power/total_power)
  feat = np.array(feat)
  names = [f'PSD_{lf}-{hf}' for lf, hf in FREQ_CUTS]
  return feat


# Pendiente espectral
def spectral_slope (data):

  fs,cough = data
  b1=0
  b2=8000
  s = np.absolute(np.fft.fft(cough))
  s = s[:s.shape[0]//2]
  muS = np.mean(s)
  f = np.linspace(0,fs/2,s.shape[0])
  muF = np.mean(f)

  bidx = np.where(np.logical_and(b1 <= f, f <= b2))
  slope = np.sum(((f-muF)*(s-muS))[bidx]) / np.sum((f[bidx]-muF)**2)
  return slope


# Chroma feature
def chroma (data):
  fs,cough = data
  chromagram = librosa.feature.chroma_stft(cough, sr=fs)
  n_chorma = 12
  chromagram_mean = chromagram.mean(axis=1)
  chromagram_std = chromagram.std(axis=1)
  chromagram = np.append(chromagram_mean,chromagram_std)
  return chromagram


# Frecuencia dominante

def Dominant_Frequency(data):
  fs,cough = data
  cough_fortan = np.asfortranarray(cough)
  freqs, psd = signal.welch(cough_fortan)
  DF = freqs[np.argmax(psd)]
  return  DF

header = 'filename zero_crossing_rate Kurtosis spectral_centroid rolloff std creast_factor RMS_power spectral_bandwidth spectral_slope  Dominant_Frequency'

for i in range (13):
  header += f' MFCC_mean{i}'

for i in range (13):
  header += f' MFCC_std{i}'

for i in range (8):
  header += f' PSD{i}'

for i in range (12):
  header += f' chroma_mean{i}'

for i in range (12):
  header += f' chroma_std{i}'

header += ' label'
header = header.split()


file_1 = open('dataset.csv', 'w', newline='')
with file_1:
    writer = csv.writer(file_1)
    writer.writerow(header)

path = "/content/Status/Healthy/"

dirs = os.listdir(path)
for file in dirs:
   dir = path + file
   print(dir)
   x,fs = librosa.load(dir, sr=None)
   senal_recortada = cut_signal(x,fs)
   try:
     new_signal, fs_new = preprocesamiento(senal_recortada)
     data = (fs_new,new_signal)
    
     #Extraccion de caracteristicas
     feature_mfcc = MFCC(data) # Espectrograma de mel 
     feature_ZCR = ZCR(data) # Zero crossing rate
     feature_Kurtosis = kurtosis(data) # Kurtosis
     feature_sc = spectral_cen(data) # Centroide espectral
     feature_rolloff = roll_off(data) # Roll off spectral
     feautre_std = std(data) # Desviacion estandar
     feautre_creast_factor = factor_cresta(data)# Factor de cresta
     feature_rms = RMS(data) # RMS Power
     feature_psd = PSD(data)# Densidad espectral de potencia
     feautre_spectral_bandwidth = spectral_bandwidth(data)# Ancho de banda espectral
     feature_spectral_slope = spectral_slope(data) # Pendiente espectral
     feature_chroma = chroma(data) # Chroma feature
     feautre_doninant_frequency = Dominant_Frequency(data) # Frecuencia dominante

     to_append = f'{file} {feature_ZCR} {feature_Kurtosis} {feature_sc} {feature_rolloff} {feautre_std} {feautre_creast_factor} {feature_rms} {feautre_spectral_bandwidth} {feature_spectral_slope} {feautre_doninant_frequency}'

     for i in range(26):
       to_append += f' {feature_mfcc[i]}' 
   
     for i in range(8):
       to_append += f' {feature_psd[i]}'
    
     for i in range (24):
       to_append += f' {feature_chroma[i]}'

     to_append += f' {1}'

     file_1 = open('dataset.csv', 'a', newline='')
     with file_1:
       writer = csv.writer(file_1)
       writer.writerow(to_append.split())


   except:
     "Feature extraction fails when the audio is completely silent"

path = "/content/Status/Covid-19/"

dirs = os.listdir(path)
for file in dirs:
   dir = path + file
   x,fs = librosa.load(dir, sr=None)
   senal_recortada = cut_signal(x,fs)
   try:
     new_signal, fs_new = preprocesamiento(senal_recortada)
     data = (fs_new,new_signal)
     #Extraccion de caracteristicas
     feature_mfcc = MFCC(data) # Espectrograma de mel 
     feature_ZCR = ZCR(data) # Zero crossing rate
     feature_Kurtosis = kurtosis(data) # Kurtosis
     feature_sc = spectral_cen(data) # Centroide espectral
     feature_rolloff = roll_off(data) # Roll off spectral
     feautre_std = std(data) # Desviacion estandar
     feautre_creast_factor = factor_cresta(data)# Factor de cresta
     feature_rms = RMS(data) # RMS Power
     feature_psd = PSD(data)# Densidad espectral de potencia
     feautre_spectral_bandwidth = spectral_bandwidth(data)# Ancho de banda espectral
     feature_spectral_slope = spectral_slope(data) # Pendiente espectral
     feature_chroma = chroma(data) # Chroma feature
     feautre_doninant_frequency = Dominant_Frequency(data) # Frecuencia dominante

     to_append = f'{file} {feature_ZCR} {feature_Kurtosis} {feature_sc} {feature_rolloff} {feautre_std} {feautre_creast_factor} {feature_rms} {feautre_spectral_bandwidth} {feature_spectral_slope} {feautre_doninant_frequency}'

     for i in range(26):
       to_append += f' {feature_mfcc[i]}' 
   
     for i in range(8):
       to_append += f' {feature_psd[i]}'
    
     for i in range (24):
       to_append += f' {feature_chroma[i]}'

     to_append += f' {0}'

     file_1 = open('dataset.csv', 'a', newline='')
     with file_1:
       writer = csv.writer(file_1)
       writer.writerow(to_append.split())


   except:
     "Feature extraction fails when the audio is completely silent"

# Creacion de matriz X y vector dependiente y

X = df.iloc[:,1:69].values
y = df.iloc[:,69].values
samples_to_extract_covid = df.label.to_numpy()
np.count_nonzero(samples_to_extract_covid == 1)

# Lectura del dataset creado con las caracteristicas
df = pd.read_csv('/content/dataset.csv')

# Creacion de matriz X y vector dependiente y

X = df.iloc[:,1:69].values
y = df.iloc[:,69].values

# Entrenamiento con valores de PCA:

# Libreria de PCA y escalizacion de los datos
sc = StandardScaler()
pca = PCA(n_components=56)

# Implementacion de SMOTE y particion del conjunto de datos: prueba y test
X_resampled, y_resampled = SMOTE().fit_resample(X, y)
X_train , X_test, y_train, y_test = train_test_split(X_resampled,y_resampled, test_size = 0.25, random_state = 0) 

# Escalizacion de los datos
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Implementacion de PCA
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Entrenamiento del algoritmo
svc_linear = SVC(C = 10, kernel = 'rbf', random_state=0)
svc_linear.fit(X_train,y_train)

# Metricas de evaluacion del algoritmo
train_score = round(svc_linear.score(X_train,y_train),3)
TN,FP,FN,TP = confusion_matrix(y_test,svc_linear.predict(X_test)).ravel()
test_score = round((TP + TN) / (TP + TN + FN + FP),3)
coeff_Matthews = round(((TP*TN)-(FP*FN))/(np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))),3)
F1_score = round((2*TP)/(2*TP+FP+FN),3)

print('test score:',test_score,'Coeficiente de Matthews:',coeff_Matthews,'F1 score:',F1_score,'train score:',train_score)
plot_confusion_matrix(svc_linear, X_test, y_test,cmap = 'Blues') 
plt.title('Covid-19 = 0      Healthy = 1')
plt.show()
