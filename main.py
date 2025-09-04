# %% Importing all necessary libraries
import librosa
import matplotlib.pyplot as plt
import librosa.display
import sklearn
import datetime
import keras
import statistics 
import random
from random import randint
from random import seed
from keras import models
from keras import layers
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from scipy import signal
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.collections import PolyCollection
import numpy as np
import csv
import pandas as pd
import math
import os


FEATURE_ORDER_DEFAULT = [
    'spectral_centroid',
    'rolloff',
    'zero_crossing_rate',
    'mfcc',
    'chroma',
    'bpm',
    'harmonic_energy',
    'percussive_energy',
]

#%% Defining the path to the genres folder

genres_path = "/Users/gianlucaproiettidemarchis/Downloads/genres"


## Extracting the first song 

##### Step 1. Import the song file
print("This song is 30 seconds long.")
file_path = os.path.join(genres_path, "blues", "blues.00030.wav")

## Load an audio file as a floating point time series.
x , sampling_rate = librosa.load(file_path)


print("The sampling rate of this file is: {0}. It represents how many times ".format(sampling_rate) +
      "per second the audio is sampled.")
print("The value \"x\" represents the wave form of a song, which contains {0} slices.".format(x.shape[0]))
print("The approximate duration of the song is: {0}. Which is what we expect since all songs are 30 seconds long.".format(len(x)/sampling_rate))

## Plot the waveform of the loaded file
plt.figure(figsize=(20, 6))
plt.title("Figure 1: Waveform of a raw audio file",fontdict={'fontsize':20})
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
librosa.display.waveshow(x, sr=sampling_rate)
plt.show()

# %% Apply Short-Time Fourier Transform (STFT) to Decompose Wave Form to base frequencies


wav_form = librosa.stft(x)
S_spectogram, phase_spectogram = librosa.magphase(librosa.stft(y=x))

## Plot the spectrogram of the STFT representation
wav_form_db = librosa.amplitude_to_db(abs(wav_form))
plt.figure(figsize=(16,9))
plt.title("Figure 3: Spectrographic representation of the waveform from fig 1.",fontdict={'fontsize':20})
plt.xlabel("Time(s)")
plt.ylabel("Frequency(Hz)")
librosa.display.specshow(wav_form_db, sr=sampling_rate, x_axis='time', y_axis='log')
plt.colorbar()
plt.inferno()







# %% Feature 1: Zero-Crossing Rate
## This represents how many times the frequency passes the "0" x-axis in either negative or positive direction.
m0 = 10000
m1 = 10100
plt.figure(figsize=(14,5))
plt.title("Figure 6: ZCR for a 4.5 ms frame",fontdict={'fontsize':20})
plt.xlabel("Slices")
plt.ylabel("Amplitude")
plt.plot(x[m0:m1])
plt.grid()
zero_crossing_sample = librosa.zero_crossings(x[m0:m1], pad=False)
print("From the image below which represents a small slice of time. The number of crossings " +
     "0 is {0} \n".format(sum(zero_crossing_sample)))
# %% Zero Crossing feature applied to entire song
zero_crossings = librosa.zero_crossings(x, pad=False)
print("When this idea of zero crossings is extended to the entire 30 second portion of a song, this song has " +
      "{0} zero crossings".format(sum(zero_crossings)))
# %% Feature 2: Spectral Centroid


n0 = 100000
n1 = 300000
x_sample = x[n0:n1]

spectral_centroids_sample = librosa.feature.spectral_centroid(y=x_sample, sr=sampling_rate)[0]
spectral_centroids_sample.shape

frames = range(len(spectral_centroids_sample))
t = librosa.frames_to_time(frames)

# Normalize
def normalize(x_sample, axis=0):
    return sklearn.preprocessing.minmax_scale(x_sample, axis=axis)
plt.figure(figsize=(14,5))
librosa.display.waveshow(x_sample, sr=sampling_rate, alpha=0.4)
plt.title("Figure 7: Spectral centroid overlaid onto waveform",fontdict={'fontsize':20})
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
plt.plot(t, normalize(spectral_centroids_sample), color='r')


S_sample, phase_sample = librosa.magphase(librosa.stft(y=x_sample))
plt.figure(figsize = (14,5))
plt.title("Figure 8: Spectral centroid overlaid onto spectrogram",fontdict={'fontsize':20})
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
librosa.display.specshow(librosa.amplitude_to_db(S_sample, ref=np.max),y_axis='hz', x_axis='time')
plt.plot(t,spectral_centroids_sample, lw = 5.0, color='b')
# %% Full song taken as spectral_centroid

spectral_centroids = librosa.feature.spectral_centroid(y=x, sr=sampling_rate)[0]
# %% Feature 3: Spectral Roll-off

## Spectral Roll Percent at 85%

spectral_rolloff = librosa.feature.spectral_rolloff(y=x+0.01, sr=sampling_rate, roll_percent=0.85)[0]

frames = range(len(spectral_rolloff))
t = librosa.frames_to_time(frames)

plt.figure(figsize=(16,5))
plt.title("Figure 9: Spectral rolloff @ 85% overlaid onto a waveform",fontdict={'fontsize':20})
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
librosa.display.waveshow(x, sr=sampling_rate, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='darkorange')


plt.figure(figsize = (16,9))
plt.subplot(2,1,1)
plt.title("Figure 10.1: Spectral rolloff @ 85% overlaid onto a spectrogram[linear]",fontdict={'fontsize':20})
librosa.display.specshow(librosa.amplitude_to_db(S_spectogram, ref=np.max),y_axis='hz', x_axis='time')
plt.plot(t,spectral_rolloff, color='darkorange', lw=3)
plt.subplot(2,1,2)
plt.title("Figure 10.2: Spectral rolloff @ 85% overlaid onto a spectrogram[log]",fontdict={'fontsize':20})
librosa.display.specshow(librosa.amplitude_to_db(S_spectogram, ref=np.max),y_axis='log', x_axis='time')
plt.plot(t,spectral_rolloff, color='darkorange', lw=3)
plt.show()
# %% Spectral Roll Percent at 45%
spectral_rolloff_2= librosa.feature.spectral_rolloff(y=x+0.01, sr=sampling_rate, roll_percent=0.45)[0]
plt.figure(figsize=(16,5))
plt.title("Figure 11: Spectral rolloff @ 45% overlaid onto a waveform",fontdict={'fontsize':20})
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
librosa.display.waveshow(x, sr=sampling_rate, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff_2), color='y')

plt.figure(figsize = (16,9))
plt.subplot(2,1,1)
plt.title("Figure 12.1: Spectral rolloff @ 45% overlaid onto a spectrogram[linear]",fontdict={'fontsize':20})
librosa.display.specshow(librosa.amplitude_to_db(S_spectogram, ref=np.max),y_axis='hz', x_axis='time')
plt.plot(t,spectral_rolloff_2, color='y', lw=3)
plt.subplot(2,1,2)
plt.title("Figure 12.2: Spectral rolloff @ 45% overlaid onto a spectrogram[log]",fontdict={'fontsize':20})
librosa.display.specshow(librosa.amplitude_to_db(S_spectogram, ref=np.max),y_axis='log', x_axis='time')
plt.plot(t,spectral_rolloff_2, color='y', lw=3)
plt.show()
# %% Spectral Roll Percent at 10%
spectral_rolloff_3= librosa.feature.spectral_rolloff(y=x+0.01, sr=sampling_rate, roll_percent=0.1)[0]
plt.figure(figsize=(16,5))
plt.title("Figure 13: Spectral rolloff @ 10% overlaid onto a waveform",fontdict={'fontsize':20})
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
librosa.display.waveshow(x, sr=sampling_rate, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff_2), color='g')

plt.figure(figsize = (16,9))

plt.subplot(2,1,1)
plt.title("Figure 14.1: Spectral rolloff @ 10% overlaid onto a spectrogram[linear]",fontdict={'fontsize':20})
librosa.display.specshow(librosa.amplitude_to_db(S_spectogram, ref=np.max),y_axis='hz', x_axis='time')
plt.plot(t,spectral_rolloff_3, color='g', lw=3)
plt.subplot(2,1,2)
plt.title("Figure 14.2: Spectral rolloff @ 10% overlaid onto a spectrogram[log]",fontdict={'fontsize':20})
librosa.display.specshow(librosa.amplitude_to_db(S_spectogram, ref=np.max),y_axis='log', x_axis='time')
plt.plot(t,spectral_rolloff_3, color='g', lw=3)
plt.show()
# %% Plotting the 3 different rolloffs percentages on the same graph both on a linear hz scale and a log hz scale
plt.figure(figsize = (16,9))
plt.subplot(2,1,1)
plt.title("Figure 15.1: Spectral rolloff values at 85/45/10 collectively overlaid onto a spectrogram[linear]",fontdict={'fontsize':20})
librosa.display.specshow(librosa.amplitude_to_db(S_spectogram, ref=np.max),y_axis='hz', x_axis='time')
plt.plot(t,spectral_rolloff, color='darkorange', lw=2.5)
plt.plot(t,spectral_rolloff_2, color='y', lw=2.5)
plt.plot(t,spectral_rolloff_3, color='g', lw=2.5)
#plt.subplot(2,1,2)
plt.title("Figure 15.2: Spectral rolloff values at 85/45/10 collectively overlaid onto a spectrogram[log]",fontdict={'fontsize':20})
librosa.display.specshow(librosa.amplitude_to_db(S_spectogram, ref=np.max),y_axis='log', x_axis='time')
plt.plot(t,spectral_rolloff, color='darkorange', lw=2.5)
plt.plot(t,spectral_rolloff_2, color='y', lw=2.5)
plt.plot(t,spectral_rolloff_3, color='g', lw=2.5)
plt.show()
# %% Feature 4: Mel-Frequency Cepstral Coefficients (MFCC)
## This feature is a way to represent human auditory sounds

S = librosa.feature.melspectrogram(y=x, sr=sampling_rate, n_mels=128)

# Converti in decibel per visualizzazione
S_db = librosa.power_to_db(S, ref=np.max)

# Mostra lo spettrogramma
plt.figure(figsize=(16, 9))
librosa.display.specshow(S_db, sr=sampling_rate, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Mel bins)")
#plt.title("Mel Spectrogram")
plt.show()

mfccs = librosa.feature.mfcc(y=x, sr=sampling_rate)
mfccs_delta = librosa.feature.delta(mfccs)
mfccs_delta2 = librosa.feature.delta(mfccs, order=2)

plt.figure(figsize=(16,9))
plt.subplot(3,1,1)
plt.title("MFCC Distribution",fontdict={'fontsize':16})
plt.xlabel("Time(s)")
plt.ylabel("Bins")
librosa.display.specshow(mfccs, sr=sampling_rate, x_axis='time')
plt.colorbar()

plt.subplot(3,1,2)
plt.title("Delta MFCCs", fontdict={'fontsize':16})
librosa.display.specshow(mfccs_delta, sr=sampling_rate, x_axis='time')
plt.colorbar()

plt.subplot(3,1,3)
plt.title("Delta-Delta MFCCs", fontdict={'fontsize':16})
librosa.display.specshow(mfccs_delta2, sr=sampling_rate, x_axis='time')
plt.colorbar()

plt.tight_layout()
plt.show()


# %% Feature 5: Chroma Frequencies

chromagram = librosa.feature.chroma_stft(y=x, sr=sampling_rate )
plt.figure(figsize=(16,9))
plt.title("Figure 17: Chroma Frequency Distribution",fontdict={'fontsize':20})
plt.xlabel("Time(s)")
plt.ylabel("Semitones")
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma')
plt.colorbar()
plt.magma()

# %% Feature 6: BPM estimation

tempo, beats = librosa.beat.beat_track(y=x, sr=sampling_rate)
beat_times = librosa.frames_to_time(beats, sr=sampling_rate)
plt.figure(figsize=(16,9))
plt.title("Figure 18: BPM estimation",fontdict={'fontsize':20})
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
librosa.display.waveshow(x, sr=sampling_rate, alpha=0.4)
plt.vlines(beat_times, ymin=min(x), ymax=max(x), color='r', linestyle='--', label='Beats')
plt.tight_layout()
plt.show()
# %% Harmonic and percussive separation
harmonic, percussive = librosa.effects.hpss(x)
plt.figure(figsize=(16,9))

plt.subplot(3, 1, 1)
librosa.display.waveshow(x, sr=sampling_rate, alpha=0.4)
plt.title('Original waveform')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 2)
librosa.display.waveshow(harmonic, sr=sampling_rate, color='g', alpha=0.4)
plt.title('Harmonic component')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 3)
librosa.display.waveshow(percussive, sr=sampling_rate, color='r', alpha=0.4)
plt.title('Percussive component')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
  

# %%
## Calculates how many sample points each file will have. The number of sample points will depend on the length of the
## sample chosen and we have decided to take 5, 10, and 20 second samples of each song in order to compare how they
## perform. Since each is sampled at the same rate, the longer samples will have more points.
#second_5=math.floor((5*22050)/512)
#second_10=math.floor((10*22050)/512)
#second_20=math.floor((20*22050)/512)


# %% csvCreatorFiveFeat, csvAppendor, Feature Extraction
## Create a csv file into which the 5 features will be written. The file will include a column for the filename,
## genre, and clomuns for all the features.

def csvCreatorSevenFeat (file_name):
    header_elements =["filename ",
                      "spectral_centroid ",
                      "rolloff ",
                      "zero_crossing_rate",
                      ]
    pitch = ["c", "c#", "d", "d#","e", "f", "f#", "g", "g#", "a", "a#", "b"]
    header=''
    for l in range (1, 21):
        header_elements.append(f' mfcc{l}')
    for l in range(1, 21):
        header_elements.append(f'mfcc_delta{l}')
    for l in range(1, 21):
        header_elements.append(f'mfcc_delta2_{l}')
    for i in range (1,13):
        header_elements.append(f"chroma_pitch_{pitch[i-1]}")

    header_elements.append('bpm') 
    header_elements.append('harmonic_energy')
    header_elements.append('percussive_energy')
    header_elements.append('genre')
    file = open(file_name,'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header_elements)



## CSV Appendor Method for adding the extrated music feautres to the CSV File. It takes as input each of the features
## as well as the name of the CSV file they will be added to, and then appends each.
def csvAppendor (csvFile, zcr, spectral_centroids, spectral_rolloff, mfccs,mfccs_delta, mfccs_delta2, chroma, bpm, harmonic, percussive, song_file_path, genre):
    harmonic_energy = np.mean(harmonic ** 2)
    percussive_energy = np.mean(percussive ** 2)

    to_append =f'{song_file_path} {np.mean(spectral_centroids)} {np.mean(spectral_rolloff)} {np.mean(zcr)}'
    for e in mfccs:
        to_append += f' {np.mean(e)}'
    # MFCC delta
    for e in mfccs_delta:
        to_append += f' {np.mean(e)}'
    # MFCC delta-delta
    for e in mfccs_delta2:
        to_append += f' {np.mean(e)}'
    for h in chroma:
        to_append += f' {np.mean(h)}'
    bpm_value = bpm[0] if isinstance(bpm, np.ndarray) else bpm    
    to_append += f' {float(bpm_value)} {harmonic_energy} {percussive_energy}'

    to_append += " "
    to_append += genre
    file = open(csvFile, 'a', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())



## Feature Extraction Method.
def featureExtractor (songFilePath, startTime, duration,rollPercent):
    x , sampling_rate = librosa.load(songFilePath, offset= startTime, duration=duration)
    zcr = librosa.zero_crossings(x)
    spectral_centroids = librosa.feature.spectral_centroid(y=x, sr=sampling_rate)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=x+0.01, sr=sampling_rate, roll_percent=rollPercent)[0]
    mfccs = librosa.feature.mfcc(y=x, sr=sampling_rate)
    mfccs_delta = librosa.feature.delta(mfccs)       # prima derivata
    mfccs_delta2 = librosa.feature.delta(mfccs, order=2)  # seconda derivata
    chroma = librosa.feature.chroma_stft(y=x, sr=sampling_rate)
    bpm, _ = librosa.beat.beat_track(y=x, sr=sampling_rate)
    harmonic, percussive = librosa.effects.hpss(x)

    return zcr, spectral_centroids, spectral_rolloff, mfccs, mfccs_delta, mfccs_delta2, chroma, bpm, harmonic, percussive
# %% Extract features for all songs
## Call the csvCreatorSevenFeat function to create the CSV files for the 5, 10, and 20 second samples

csvCreatorSevenFeat('feature_extraction_5sec.csv')
csvCreatorSevenFeat('feature_extraction_10sec.csv')
csvCreatorSevenFeat('feature_extraction_20sec.csv')


## Create a list of the 10 genres in the dataset to iterate through
genres = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]

file_path_b="/Users/gianlucaproiettidemarchis/Downloads/genres/"

## Iterate through the list of genres and for each one, open the folder containing its songs and for every song,
## create a 5, 10, and 20 second sample and write those samples into their respective CSV file.
print("Start")
for x in genres:
    file_path_g= file_path_b + x +'/'+x +'.000'

    for y in range(0, 100):
        y_string = str(y)
        file_path_s = file_path_g + y_string.rjust(2,'0') + '.wav'

        #extracting features from 5 seconds
        zcr, spectral_centroids, spectral_rolloff, mfccs, mfccs_delta, mfccs_delta2, chroma, bpm, harmonic, percussive = featureExtractor(file_path_s, 12.5,5,0.85)
        csvAppendor('feature_extraction_5sec.csv',zcr, spectral_centroids, spectral_rolloff, mfccs, mfccs_delta, mfccs_delta2, chroma, bpm, harmonic, percussive, file_path_s[file_path_s.rfind('/')+1:], x)
         #extracting features from 10 seconds
        zcr, spectral_centroids, spectral_rolloff, mfccs, mfccs_delta, mfccs_delta2, chroma, bpm, harmonic, percussive = featureExtractor(file_path_s, 10,10,0.85)
        csvAppendor('feature_extraction_10sec.csv',zcr, spectral_centroids, spectral_rolloff, mfccs, mfccs_delta, mfccs_delta2, chroma, bpm, harmonic, percussive, file_path_s[file_path_s.rfind('/')+1:], x)
         #extracting features from 20 seconds
        zcr, spectral_centroids, spectral_rolloff, mfccs, mfccs_delta, mfccs_delta2, chroma, bpm, harmonic, percussive = featureExtractor(file_path_s, 5,20,0.85)
        csvAppendor('feature_extraction_20sec.csv',zcr, spectral_centroids, spectral_rolloff, mfccs, mfccs_delta, mfccs_delta2, chroma, bpm, harmonic, percussive, file_path_s[file_path_s.rfind('/')+1:], x)
    print(x)
print("End")

# %% Read the data from the features CSV
# Read the data from the features CSV into a pandas DataFrame and then display the first 110 rows
X = pd.read_csv("feature_extraction_10sec.csv")
X.head(110)


# %% lookup e feature group
def _build_column_lookup(cols):
    """
    Ritorna un dizionario {nome_norm: nome_originale} per gestire spazi finali.
    Normalizza in lower-case e strip per il lookup, ma conserva il nome originale.
    """
    lookup = {}
    for c in cols:
        norm = c.strip().lower()
        lookup[norm] = c 
    return lookup




def _expand_feature_group(group_name, lookup):
    """
    Dato un gruppo logico e la mappa lookup, ritorna la lista di nomi colonna originali
    corrispondenti a quel gruppo. Se una colonna manca, viene ignorata.
    """
    cols = []

    if group_name == 'spectral_centroid':
        if 'spectral_centroid' in lookup: cols.append(lookup['spectral_centroid'])

    elif group_name == 'rolloff':
        if 'rolloff' in lookup: cols.append(lookup['rolloff'])

    elif group_name == 'zero_crossing_rate':
        if 'zero_crossing_rate' in lookup: cols.append(lookup['zero_crossing_rate'])

    elif group_name == 'mfcc':
        for i in range(1, 21):
            key = f'mfcc{i}'
            if key in lookup:
                cols.append(lookup[key])
                    # MFCC delta
        for i in range(1, 21):
            key = f'mfcc_delta{i}'
            if key in lookup:
                cols.append(lookup[key])
         # MFCC delta-delta
        for i in range(1, 21):
            key = f'mfcc_delta2_{i}'
            if key in lookup:
                cols.append(lookup[key])

    elif group_name == 'chroma':
        pitch_names = ['c','c#','d','d#','e','f','f#','g','g#','a','a#','b']
        for p in pitch_names:
            key = f'chroma_pitch_{p}'
            if key in lookup:
                cols.append(lookup[key])

    elif group_name == 'bpm':
        if 'bpm' in lookup: cols.append(lookup['bpm'])

    elif group_name == 'harmonic_energy':
        if 'harmonic_energy' in lookup:
            cols.append(lookup['harmonic_energy'])
        if 'percussive_energy' in lookup:
            cols.append(lookup['percussive_energy'])

    return cols



# %%

def read_data(file_name):
    data = pd.read_csv(file_name)
    return data

def create_genre_list(data):
    genre_list = data.iloc[:, -1]
    return genre_list

def get_y_value(data):
    genre_list = create_genre_list(data)
    encoder = preprocessing.LabelEncoder()
    y = encoder.fit_transform(genre_list)
    return y


def get_x_value(data, num_features=5, feature_order=None):
    """
    Seleziona le feature *concettuali* (gruppi) in base a num_features.
    Applica StandardScaler e restituisce:
      X  -> array normalizzato (n_samples, n_columns_selezionate)
      used_columns -> lista dei nomi colonna effettivamente usati (per logging)
      used_groups  -> lista dei gruppi logici usati (per stampa)
    """
    if feature_order is None:
        feature_order = FEATURE_ORDER_DEFAULT

    # Copia per sicurezza
    df = data.copy()

    # Drop label e non-feature
    # Nota: nel CSV hai 'filename ' con spazio finale -> controlliamo entrambe le varianti
    for col_candidate in ['filename ', 'filename']:
        if col_candidate in df.columns:
            df = df.drop(columns=[col_candidate])
            break

    if 'genre' in df.columns:
        df = df.drop(columns=['genre'])

    # DEBUG: stampo le colonne rimanenti
    #print("DEBUG - Colonne dopo drop:", df.columns.tolist())

    # Mappa normalizzata -> colonna originale
    lookup = _build_column_lookup(df.columns)

    # DEBUG: stampo lookup
    #print("DEBUG - Lookup colonne per gruppo:", lookup)

    # Determina i gruppi da usare
    groups_to_use = feature_order[:num_features]

    # DEBUG: stampo gruppi selezionati
    #print("DEBUG - Gruppi selezionati:", groups_to_use)

    # Espandi in colonne reali
    selected_cols = []
    for g in groups_to_use:
        cols = _expand_feature_group(g, lookup)
        # DEBUG: stampo colonne estratte da ogni gruppo
        #print(f"DEBUG - Gruppo '{g}' -> colonne {cols}")
        selected_cols.extend(cols)

    # DEBUG: stampo tutte le colonne selezionate
    #print("DEBUG - Colonne finali selezionate:", selected_cols)

    if not selected_cols:
        raise ValueError("Nessuna colonna selezionata: controlla i nomi nel CSV.")

    # Estrai i dati (mantieni ordine)
    df_sel = df[selected_cols]

    # Scala
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(df_sel.to_numpy(dtype=float))

    return X, selected_cols, groups_to_use



def get_feature_file_names():
    return ["feature_extraction_5sec.csv",
            "feature_extraction_10sec.csv",
            "feature_extraction_20sec.csv"]
# %% Neural network
def create_and_run_nn(X_train, X_test, y_train, y_test):
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    hist = model.fit(X_train,
                    y_train,
                    epochs=20,
                    verbose=0,
                    batch_size=32)

    # Get Accuracy
    test_loss, test_acc = model.evaluate(X_test,y_test)

    #predictions
    y_pred_probs = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    
    return test_acc, test_loss, y_test, y_pred_classes, hist.history

def train_and_eval(file, num_iterations=1, num_features = 5, feature_order=None):
    i = 0
    acc_list = []
    all_y_test = []
    all_y_pred = []
    used_cols, used_groups = None, None


    
    while i < num_iterations:
        data = read_data(file)
        y = get_y_value(data)
        
        X, used_cols, used_groups = get_x_value(data, num_features=num_features, feature_order=feature_order)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        test_acc_result, _ , y_test, y_pred, _  = create_and_run_nn(X_train, X_test, y_train, y_test)
        acc_list.append(test_acc_result)
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)
        i += 1
    return acc_list, all_y_test, all_y_pred, used_cols, used_groups

# %% Get statistics
# Gathers statistics on test accuracy list, and passes them to a print function
def get_statistics(acc_list, song_length=5, num_features=0, csv_path=None, all_y_test=None, all_y_pred=None):
    mean = statistics.mean(acc_list)
    min_res = min(acc_list)
    max_res = max(acc_list)
    data = None
    groups, cols = None, None
    if csv_path is not None:
        try:
            data = pd.read_csv(csv_path)
            _, cols, groups = get_x_value(data, num_features=num_features)
        except Exception as e:
            print(f"Could not read CSV: {e}")
    print_results(song_length, mean, min_res, max_res, used_groups=groups, used_columns=cols, all_y_test=all_y_test, all_y_pred=all_y_pred)






def print_results(song_length, mean, min_res, max_res, used_groups=None, used_columns=None, all_y_test=None, all_y_pred=None):
        if not used_groups:
            print(f"Baseline file with {song_length} second song clips, scored a mean test accuracy of: {mean:.4f}")
        else:
            print(f"{len(used_groups)} Feature-group file with {song_length} second song clips, scored a mean test accuracy of: {mean:.4f}")
            print(f"Groups used: {', '.join(used_groups)}")

        if used_columns:
            print(f"Columns used ({len(used_columns)} total): {', '.join(used_columns)}")
        else:
            print("Feature columns not available.")

        print(f"With a Min score of: {min_res:.4f}  ---  Max score of: {max_res:.4f}")
        print("---------------------------------------------------------------------------")

         # Se abbiamo le predizioni, stampiamo confusion matrix e report
        if all_y_test is not None and all_y_pred is not None:
            genre_names = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
            cm = confusion_matrix(all_y_test, all_y_pred)
            cm_df = pd.DataFrame(cm, index=genre_names, columns=genre_names)
            print("Confusion Matrix:\n", cm_df)
            report = classification_report(all_y_test, all_y_pred, target_names=genre_names)
            print("\nClassification Report:\n", report)

                # --- Plot and save ---
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=genre_names)
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)

        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.pdf", bbox_inches="tight")  # salva PDF
        plt.savefig("confusion_matrix.png", bbox_inches="tight")  # salva anche PNG
        plt.close()
             # --- Classification Report ---
              # --- Classification Report ---
        report_dict = classification_report(all_y_test, all_y_pred, target_names=genre_names, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()

        # plot come tabella
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        ax.axis('tight')

        # crea tabella
        table = ax.table(cellText=report_df.round(2).values,
                         colLabels=report_df.columns,
                         rowLabels=report_df.index,
                         cellLoc='center', loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.2)

        plt.title("Classification Report")
        plt.savefig("classification_report.pdf", bbox_inches="tight")
        plt.savefig("classification_report.png", bbox_inches="tight")
        plt.close()



# %%
# Generate test accuracy results of 3 features for num_iteration times
def run_nn_main_3_ft(num_iterations=1):
    files_5f = get_feature_file_names()
    
    durations = [5, 10, 20]  # secondi
    all_stats = {}  # qui accumuliamo statistiche separate
    all_acc_global = []
    all_y_test_global = []
    all_y_pred_global = []

    for idx, file in enumerate(files_5f):
        duration = durations[idx]
        acc_list, y_test, y_pred, _, _ = train_and_eval(file, num_iterations, 3, ['zero_crossing_rate', 'mfcc', 'harmonic_energy'])

        # accumula statistiche separate per durata
        all_stats[duration] = {
            "acc_list": acc_list,
            "y_test": y_test,
            "y_pred": y_pred,
            "file": file
        }

        # accumula statistiche globali
        all_acc_global.extend(acc_list)
        all_y_test_global.extend(y_test)
        all_y_pred_global.extend(y_pred)

    # Stampa statistiche separate per durata
    for duration in durations:
        stats = all_stats[duration]
        print(f"\n--- Statistiche per clip da {duration} secondi ---")
        get_statistics(stats["acc_list"], song_length=duration, num_features=3, 
                       csv_path=stats["file"], all_y_test=stats["y_test"], all_y_pred=stats["y_pred"])

    # Stampa statistiche aggregate globali
    print("\n--- Statistiche aggregate su tutte le durate ---")
    get_statistics(all_acc_global, song_length=0, num_features=3,csv_path=stats["file"],
                   all_y_test=all_y_test_global, all_y_pred=all_y_pred_global)




run_nn_main_3_ft(20)

# %%
 # Generate test accuracy results of 5 features for num_iteration times
def run_nn_main_5_ft(num_iterations=1):
    files_5f = get_feature_file_names()
    durations = [5, 10, 20]  # secondi
    all_stats = {}  # qui accumuliamo statistiche separate
    all_acc_global = []
    all_y_test_global = []
    all_y_pred_global = []

    for idx, file in enumerate(files_5f):
        duration = durations[idx]
        acc_list, y_test, y_pred, _, _ = train_and_eval(file, num_iterations, 5, FEATURE_ORDER_DEFAULT)

        # accumula statistiche separate per durata
        all_stats[duration] = {
            "acc_list": acc_list,
            "y_test": y_test,
            "y_pred": y_pred,
            "file": file
        }

        # accumula statistiche globali
        all_acc_global.extend(acc_list)
        all_y_test_global.extend(y_test)
        all_y_pred_global.extend(y_pred)

    # Stampa statistiche separate per durata
    for duration in durations:
        stats = all_stats[duration]
        print(f"\n--- Statistiche per clip da {duration} secondi ---")
        get_statistics(stats["acc_list"], song_length=duration, num_features=5, 
                       csv_path=stats["file"], all_y_test=stats["y_test"], all_y_pred=stats["y_pred"])

    # Stampa statistiche aggregate globali
    print("\n--- Statistiche aggregate su tutte le durate ---")
    get_statistics(all_acc_global, song_length=0, num_features=5,
                   csv_path=stats["file"], all_y_test=all_y_test_global, all_y_pred=all_y_pred_global)




run_nn_main_5_ft(20)

 # %%
# Generate test accuracy results of 7 features for num_iterations times
def run_nn_main_7_ft(num_iterations=1):
    files_7f = get_feature_file_names()  # funzione che restituisce i CSV per le 7 feature


    durations = [5, 10, 20]  # secondi
    all_stats = {}  # qui accumuliamo statistiche separate
    all_acc_global = []
    all_y_test_global = []
    all_y_pred_global = []

    for idx, file in enumerate(files_7f):
        duration = durations[idx]
        acc_list, y_test, y_pred, _, _ = train_and_eval(file, num_iterations, 8)

        # accumula statistiche separate per durata
        all_stats[duration] = {
            "acc_list": acc_list,
            "y_test": y_test,
            "y_pred": y_pred,
            "file": file
        }

        # accumula statistiche globali
        all_acc_global.extend(acc_list)
        all_y_test_global.extend(y_test)
        all_y_pred_global.extend(y_pred)

    # Stampa statistiche separate per durata
    for duration in durations:
        stats = all_stats[duration]
        print(f"\n--- Statistiche per clip da {duration} secondi ---")
        get_statistics(stats["acc_list"], song_length=duration, num_features=8, 
                       csv_path=stats["file"], all_y_test=stats["y_test"], all_y_pred=stats["y_pred"])

    # Stampa statistiche aggregate globali
    print("\n--- Statistiche aggregate su tutte le durate ---")
    get_statistics(all_acc_global, song_length=[5, 10, 20], num_features=8,
                   csv_path=stats["file"], all_y_test=all_y_test_global, all_y_pred=all_y_pred_global)
# Eseguiamo il metodo
run_nn_main_7_ft(20)



# %%
import itertools

def evaluate_all_feature_combinations_multiduration(files, all_feature_groups, num_iterations=1):
    """
    Esegue il training della rete per tutte le combinazioni di feature e tutte le durate,
    e restituisce la combinazione con la miglior accuracy media globale.

    Args:
        files: lista di CSV corrispondenti a durate diverse (es. 5, 10, 20 sec)
        all_feature_groups: lista di tutti i gruppi di feature disponibili
        num_iterations: numero di iterazioni per ciascuna combinazione

    Returns:
        best_combination: lista dei gruppi di feature migliori
        best_acc: miglior accuracy media su tutte le durate
        all_results: dizionario con tutte le combinazioni e le rispettive accuracy medie
    """
    best_acc = 0
    best_combination = None
    all_results = {}
    durations = [20]

    for r in range(1, len(all_feature_groups)+1):
        for comb in itertools.combinations(all_feature_groups, r):
            print(f"Testing combination: {comb}")
            acc_total = []

            # Esegui train per ogni durata
            for idx, file in enumerate(files):
                acc_list, _, _, _, _ = train_and_eval(file, num_iterations=num_iterations, feature_order=list(comb))
                acc_total.extend(acc_list)

            mean_acc = sum(acc_total)/len(acc_total)
            all_results[comb] = mean_acc
            print("Best combination of features:", best_combination)
            print("Best mean accuracy across all durations:", best_acc)

            if mean_acc > best_acc:
                best_acc = mean_acc
                best_combination = comb

            
    print("Best combination of features:", best_combination)
    print("Best mean accuracy across all durations:", best_acc)
    
    return best_combination, best_acc, all_results



