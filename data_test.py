import os
import wave
import librosa
import numpy as np
from tqdm import tqdm
import pickle as pkl
import librosa
from sklearn.preprocessing import normalize


def extract_logmel(y, sr, size=3):
    """
    extract log mel spectrogram feature
    :param y: the input signal (audio time series)
    :param sr: sample rate of 'y'
    :param size: the length (seconds) of random crop from original audio, default as 3 seconds
    :return: log-mel spectrogram feature
    """
    # normalization
    y = y.astype(np.float32)
    normalization_factor = 1 / np.max(np.abs(y))
    y = y * normalization_factor

    # random crop
    if len(y) <= size * sr:
        new_y = np.zeros((size * sr+1, ))
        new_y[:len(y)] = y
        y = new_y

    start = np.random.randint(0, len(y) - size * sr)
    y = y[start: start + size * sr]

    # extract log mel spectrogram #####
    melspectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=1024, n_mels=60)
    logmelspec = librosa.power_to_db(melspectrogram)

    return logmelspec.T


def get_wave_norm(file):
    data, framerate = librosa.load(file, sr=8000)
    return data, framerate


LABELS = ['awake', 'diaper', 'hug', 'hungry', 'sleepy', 'uncomfortable']
N_CLASS = len(LABELS)
DATA_DIR = './test'

file_glob = []

for cls_fold in tqdm(os.listdir(DATA_DIR)):

    file_pt = os.path.join(DATA_DIR, cls_fold)
    file_glob.append(file_pt)

print(len(file_glob))
print('done.')

data = {}

for file in tqdm(file_glob):

    temp = []

    raw, sr = get_wave_norm(file)
    length = raw.shape[0]
    seg = sr * 5
    for i in range((length//seg)*3+1):
        start = i * int(seg/3)
        end = start + seg
        if end > length:
            end = length
        if end - start > sr * 2:
            x = raw[start:end]
        else:
            break
        x = extract_logmel(x, sr)
        if not x.shape == (24, 60):
            x = np.resize(x, (24, 60))
            print('???')
        temp.append(x)
    data[file] = np.array(temp)

with open('data_test.pkl', 'wb') as f:
    pkl.dump(data, f)
