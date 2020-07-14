import os
import wave
import librosa
import numpy as np
from tqdm import tqdm
import pickle as pkl
import librosa
from sklearn.preprocessing import normalize
import config as cfg

def extract_logmel(y, sr, size):

    # normalization
    y = y.astype(np.float32)
    normalization_factor = 1 / np.max(np.abs(y))
    y = y * normalization_factor

    # random crop
    if len(y) <= size * sr:
        new_y = np.zeros((size * sr+1, ))
        new_y[:len(y)] = y
        y = new_y

    # extract log mel spectrogram #####
    melspectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=1024, n_mels=cfg.N_MEL)
    logmelspec = librosa.power_to_db(melspectrogram)

    return logmelspec.T


def get_wave_norm(file):
    data, framerate = librosa.load(file, sr=cfg.SR)
    return data, framerate

if __name__ == '__main__':

    DATA_DIR = './train'

    file_glob = []

    for i, cls_fold in tqdm(enumerate(cfg.LABELS)):

        cls_base = os.path.join(DATA_DIR, cls_fold)
        files = os.listdir(cls_base)
        print('{} train num:'.format(cls_fold), len(files))
        for pt in files:
            file_pt = os.path.join(cls_base, pt)
            file_glob.append((file_pt, cfg.LABELS.index(cls_fold)))

    print('done.')

    data = []

    for file, lbl in tqdm(file_glob):
        raw, sr = get_wave_norm(file)
        seg = sr * cfg.TIME_SEG
        length = raw.shape[0]
        for i in range((length//seg)*cfg.STRIDE+1):
            start = i * int(seg/cfg.STRIDE)
            end = start + seg
            if end > length:
                end = length
            if end - start > sr * 2:
                x = raw[start:end]
            else:
                break
            y = np.zeros(cfg.N_CLASS)
            y[lbl] = 1
            x = extract_logmel(x, sr, size=cfg.TIME_SEG)
            data.append((x, y))

    print(len(data))

    with open('data.pkl', 'wb') as f:
        pkl.dump(data, f)

    print(data[0][0].shape)
