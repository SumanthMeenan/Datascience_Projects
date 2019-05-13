import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import sklearn
import os
import csv
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import models #(2 use tf1, if it is using tf2)
from tensorflow.keras import layers

""" TODO: PLot Acoustic signal of an Audio file, Extract its features """

def f():
    plt.show()

"""1. Loading Audio file"""
#audio signal is a frequencies data

audio_file = 'T08-violin.wav'
audio_path = '/home/sumanthmeenan/Desktop/projects/music genre classification/T08-violin.wav'

#output - timeseries as numpy array,sampling rate (default = 22KHZ)
# input signal(audio time series)
#sample rate is the number of samples of audio carried per second, measured in Hz or kHz.

#we're loading input audio signal
time_series, sampling_rate = librosa.load(audio_path)
print('lenght of time-series array:', time_series)
print('Default sampling rate:', sampling_rate)

print(type(time_series), type(sampling_rate))
print(time_series)
print(time_series.shape, sampling_rate)

#we can change default sampling rate value
librosa.load(audio_path, sr = 44100)
librosa.load(audio_path, sr = None)


"""2. Playing Audio only in jupiter notebook"""

# from Ipython.display import display
# import IPython.display.Audio as ipd
# ipd.Audio(audio_path)
  

"""3. Visualise the audio"""
#1. waveform
# timeseries numpy array is plotted
#plot of amplitude of waveform

plt.figure(figsize= (14, 5))
librosa.display.waveplot(time_series, sr=sampling_rate)
f()

#2. spectrogram - visual representation of spectrum of frequencies of sound
#freq v/s time
#stft - short time fourier transform
X = librosa.stft(time_series)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize = (14, 5))
librosa.display.specshow(Xdb, sr = sampling_rate, x_axis = 'time', y_axis= 'hz')
f()

#convert freq librosa.output.write_wav('example.wav', x, sr)is to logarithmic one
librosa.display.specshow(Xdb, sr=sampling_rate, x_axis='time', y_axis='log')
f()

"""3. writing audio file  -  numpy array to audio file"""
librosa.output.write_wav('sample.wav', time_series, sampling_rate)

"""4. creating audio signal"""
sampling_rate1 = 22050
Time = 5.0
t = np.linspace(0, Time, int(Time*sampling_rate1), endpoint=False)
sine_wave = 0.5*np.sin(2*np.pi*220*t)
#sine wave = sound signal/ audio signal

#saving the audio
librosa.output.write_wav('tone_220.wav', t, sampling_rate1)

"""5. feature extraction"""

#1. zero-crossing rate
time_series, sampling_rate = librosa.load('/home/sumanthmeenan/Desktop/projects/music genre classification/T08-violin.wav')
librosa.display.waveplot(time_series, sr=sampling_rate)

n0 = 9000
n1 = 9100
plt.figure(figsize=(14, 5))
plt.plot(time_series[n0:n1])
plt.grid()
f()

zero_crossings = librosa.zero_crossings(time_series[n0:n1], pad=False)
print('No. of times signal is crossing zero:', sum(zero_crossings))

#2. Spectral Centroid
#calculates centre of mass for a sound is located, weighted mean of the frequencies present in the sound, 
#librosa.feature.spectral_centroid computes the spectral centroid for each frame in a signal:

spectral_centroids = librosa.feature.spectral_centroid(time_series, sr=sampling_rate)
spectral_centroids.shape
centroid_frequencies = spectral_centroids[0]
centroid_frequencies.shape
print('spectral centroid frequencies:',centroid_frequencies)

#Computing the time variable for visualization
frames = range(len(centroid_frequencies))
print('NO. of frames are:', len(frames))
t = librosa.frames_to_time(frames)
t.shape
plt.hist(t)
f()

def normalization(input, axis = 0):
    return sklearn.preprocessing.minmax_scale(input, axis = axis)

#plot spectral centroid along waveform
norm_centroid_freq = normalization(centroid_frequencies)
print('normalised centroid frequencies:', norm_centroid_freq)

librosa.display.waveplot(time_series, sr = sampling_rate)
plt.plot(t, norm_centroid_freq, color = 'r')
plt.title('spectral centroid along waveform')
#1st dimension of t and norm_centroid_Freq shld be same
f()

#3. spectral rolloff
# shape of signal which represents  the freq below which x% of total spectral energy lies
# cal. rolloff freq for each frame in signal
"""The roll-off frequency is defined for each frame as the center frequency
 for a spectrogram bin such that at least roll_percent (0.85 by default)
 of the energy of the spectrum in this frame is contained in this bin and the bins below"""

spectral_rolloff = librosa.feature.spectral_rolloff(time_series+0.01 , sr = sampling_rate)
spectral_rolloff.shape
spectral_rolloff[0].shape
#(1,775) v/s (775,)
norm_spectral_rolloff = normalization(spectral_rolloff[0])

librosa.display.waveplot(time_series, sr = sampling_rate)
plt.plot(t, norm_spectral_rolloff, color = 'r')
plt.title('spectral roll-off')
f()

"""4. Mel frequency cepstral coefficients (MFCCs)
Tells overall shape of a spectral envelope. it models the characteristics of a human voice
librosa mfcc feature cal mfccs across an audio signal"""

time_series, sampling_rate = librosa.load('/home/sumanthmeenan/Desktop/projects/music genre classification/simple-loop.wav.crdownload')
librosa.display.waveplot(time_series, sr = sampling_rate)
f()

mfcc = librosa.feature.mfcc(time_series, sr = sampling_rate)
print('Shape Of MFCCs:',mfcc.shape)
print('mfcc calculated {} MFCCS over {} frames'.format(mfcc.shape[0], mfcc.shape[1]))

# 'display mfccs'
librosa.display.specshow(mfcc, sr = sampling_rate, x_axis='time')
f()

#normalising mfcc valules 2 make each co-ef dimension 0 mean and 1 variance
scaled_mfcc = sklearn.preprocessing.scale(mfcc, axis = 1)
scaled_mfcc
scaled_mfcc.shape
scaled_mfcc[0].shape
norm_mfcc = normalization(mfcc, axis = 1)
norm_mfcc

print(mfcc.mean(axis = 0))
print(mfcc.mean(axis = 0).shape)

print(mfcc.mean(axis = 1))
print(mfcc.mean(axis = 1).shape)
print(scaled_mfcc.mean(axis = 1))
print(norm_mfcc.mean(axis = 1))

print(mfcc.var(axis = 1))
print(scaled_mfcc.var(axis = 1))
print(norm_mfcc.var(axis = 1))

"""DOUBT - NORM_MFCC MEAN = 0 AND SCALED_MFCC VAR = 1 """
librosa.display.specshow(scaled_mfcc, sr = sampling_rate, x_axis='time')
plt.title('scaled_mfcc')
f()

librosa.display.specshow(norm_mfcc, sr = sampling_rate, x_axis='time')
plt.title('norm_mfcc')
entire spectrum is projected onto 12 bins representing the 12 distinct semitones (or chroma)f()

librosa.display.specshow(mfcc, sr = sampling_rate, x_axis='time')
plt.title('mfcc')
f()

#5. Chroma frequencies
#In chroma features, entire spectrum is projected onto 12 bins representing the 12 distinct semitones (or chroma)
time_series, sampling_rate = librosa.load('/home/sumanthmeenan/Desktop/projects/music genre classification/Grand Piano.wav')

hop_length = 512
chromagram = librosa.feature.chroma_stft(time_series, sr = sampling_rate, hop_length = hop_length)
chromagram.shape
np.mean(chromagram, axis = 1).shape #Try axis = 0 v/s 1`
"""chromagram shape: np.ndarray [shape=(n_chroma, t)]
Normalized energy for each chroma bin at each frame."""

plt.figure(figsize=(15,5))
librosa.display.specshow(chromagram, sr = sampling_rate, x_axis='time',y_axis='chroma', hop_length=hop_length, cmap = 'viridis')
plt.title('chromagram')
f()

"""Use an energy (magnitude) spectrum instead of power spectrogram"""
S = np.abs(librosa.stft(time_series))
chroma = librosa.feature.chroma_stft(S=S, sr=sampling_rate)

plt.figure(figsize=(15,5))
librosa.display.specshow(chroma, sr = sampling_rate, x_axis='time',y_axis='chroma', hop_length=hop_length, cmap = 'coolwarm')
plt.title('Energy spectrum - chroma1')
f()

"""Use a pre-computed power spectrogram with a larger frame"""
S = np.abs(librosa.stft(time_series, n_fft=4096))**2
chroma = librosa.feature.chroma_stft(S=S, sr=sampling_rate)

plt.figure(figsize=(15,5))
librosa.display.specshow(chroma, sr = sampling_rate, x_axis='time',y_axis='chroma', hop_length=hop_length, cmap = 'coolwarm')
plt.title('Energy spectrum - chroma2')
f()

#6 spectral bandwidth
time_series, sampling_rate = librosa.load('/home/sumanthmeenan/Desktop/projects/music genre classification/Grand Piano.wav')
spectral_bandwidth = librosa.feature.spectral_bandwidth(time_series, sr=sampling_rate) 

"""convert .au to .wav 4 compatability with pythons wave module 4 reading audio files. 
we need ffmpeg,pydub"""

from pydub import AudioSegment
genres = os.listdir('/home/sumanthmeenan/Desktop/projects/music genre classification/genres')
for i in genres:
    au_files = os.listdir('/home/sumanthmeenan/Desktop/projects/music genre classification/genres/' + i + '/au')
    for j in au_files:
        sound = AudioSegment.from_file('/home/sumanthmeenan/Desktop/projects/music genre classification/genres/' + i + '/au/' + j, "au")
        sound.export('/home/sumanthmeenan/Desktop/projects/music genre classification/genres/' +  i + '/wav/' + str(j[:-3]) + '.wav', format="wav")

sound = AudioSegment.from_mp3('/home/sumanthmeenan/Desktop/projects/music genre classification/genres/disco/disco.00082.au')
sound.export('/home/sumanthmeenan/Desktop/projects/music genre classification/got.wav', format="wav")

"""Save spectrogram of every Audio File"""
cmap = plt.get_cmap('inferno')
plt.figure(figsize = (10,10))

genres = os.listdir('/home/sumanthmeenan/Desktop/projects/music genre classification/genres')
for i in genres:
    au_files = os.listdir('/home/sumanthmeenan/Desktop/projects/music genre classification/genres/' + i + '/au')
    for j in au_files:
        time_series,sampling_rate = librosa.load('/home/sumanthmeenan/Desktop/projects/music genre classification/genres/' + i + '/au/' + j, "au")
        plt.specgram(time_series, NFFT = 2048, Fs = 2, Fc = 0, noverlap=128,cmap  = cmap,sides='default',
         mode='default',scale='dB')
        plt.savefig('/home/sumanthmeenan/Desktop/projects/music genre classification/genres/' +  i + '/img/' + str(j[:-3]) + '.jpg')

"""Extract 5 features(MFCC, SPECTRAL CENTROID, SPECTRAL ROLL-OFF,
 Chroma Frequencies, Zero crossing rate) from each spectrogram and store it in a CSV file"""

#feature names
features = 'filename zero_crossing_rate spectral_centroid spectral_rolloff chroma rmse spectral_bandwidth '
for i in range(1, 21):
    features += 'mfcc'+str(i) + " "  #f-strings
features += 'label'
features = features.split()

""" Writing Data to CSV file """
file1 = open('/home/sumanthmeenan/Desktop/projects/music genre classification/created_data.csv', 'w')
with file1:
        writer = csv.writer(file1)
        writer.writerow(features)

genres = os.listdir('/home/sumanthmeenan/Desktop/projects/music genre classification/genres')

for i in genres:
    au_files = os.listdir('/home/sumanthmeenan/Desktop/projects/music genre classification/genres/' + i + '/au')
    for j in au_files:
        x = []
        time_series,sampling_rate = librosa.load('/home/sumanthmeenan/Desktop/projects/music genre classification/genres/' + i + '/au/' + j)
        x.append(j)
        zero_crossing_rate = librosa.zero_crossings(time_series, pad=False)
        x.append(np.mean(zero_crossing_rate))        
        spectral_centroid = librosa.feature.spectral_centroid(time_series, sr=sampling_rate) 
        x.append(np.mean(spectral_centroid))        
        spectral_rolloff = librosa.feature.spectral_rolloff(time_series+0.01 , sr = sampling_rate)
        x.append(np.mean(spectral_rolloff))        
        chroma = librosa.feature.chroma_stft(time_series, sr = sampling_rate, hop_length = 512) #chroma
        x.append(np.mean(chroma))        
        rmse = librosa.feature.rmse(time_series)        
        x.append(np.mean(rmse))        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(time_series, sr=sampling_rate) 
        x.append(np.mean(spectral_bandwidth))        
        mfcc = librosa.feature.mfcc(time_series, sr=sampling_rate)
        
        #if we use {} - dynamic values. Instead of append() we can use fstrings
        for k in mfcc:
            x.append(np.mean(k)) 
        
        x.append(i)
        print(x)

        file = open('/home/sumanthmeenan/Desktop/projects/music genre classification/created_data.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(x)

data = pd.read_csv('/home/sumanthmeenan/Desktop/projects/music genre classification/created_data.csv')
data.head()
data.shape
data1 = data.drop(data.columns[0], axis = 1)

#Label Encoding
LabelEncoder = preprocessing.LabelEncoder()
LabelEncoder.fit(data['label'])
list(LabelEncoder.classes_)
labels = LabelEncoder.transform(data['label'])

#Feature Scaling
feature_scaler = StandardScaler()
features = feature_scaler.fit_transform(np.array(data1.iloc[:, :-1], dtype = float))

#Train-Test-Split
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = 0.3)

print('shape of x_train is:', x_train.shape)
print('shape of y_train is:', y_train.shape)
print('shape of x_test is:', x_test.shape)
print('shape of y_test is:', y_test.shape)

#Applying classification Algorithms to Data

#Initialize a NN
model = models.Sequential()
#1st hidden layer has 256 nuerons, input layer has 26 nuerons(26 features in data)
model.add(layers.Dense(256, activation='relu', input_shape=(x_train.shape[1],)))
#2nd hidden layer has 128 nuerons
model.add(layers.Dense(128, activation='relu'))
#3rd hidden layer has 64 nuerons
model.add(layers.Dense(64, activation='relu'))
#output layer has 10 nuerons
model.add(layers.Dense(10, activation='softmax'))

#loss = 'sparse_categorical_crossentropy' for multiclass labels
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train,
                    y_train,
                    epochs=20,
                    batch_size=128)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print('Accuracy of test data is:', test_accuracy)

""" SPlit DATA into TRAINING + VALIDATION + TESTING """
x_val = x_train[:200]
x_train1 = x_train[200:]

y_val = y_train[:200]
y_train1 = y_train[200:]

#x_train1, y_train1 - training  data
#x_val, y_val - validation data
#x_test, y_test - testing data

history = model.fit(x_train1,
                    y_train1,
                    epochs=20,
                    batch_size=128)

test_loss, test_accuracy = model.evaluate(x_val, y_val)

#Model memorized not generalized - Overfitting
y_pred = model.predict(x_test)
y_pred[0]
print('predicted genre is:', np.argmax(y_pred[0]))
x_test[0]
print('Actual genre is:', y_test[0])

y_pred[20]
print('predicted genre is:', np.argmax(y_pred[20]))
x_test[20]
print('Actual genre is:', y_test[20])