



import numpy as np
import config as cfg
## caution:
cfg.TIME_SEG = 1.4
## data_all.py imports
import os
import wave
import librosa
import numpy as np
from tqdm import tqdm
import pickle as pkl
import librosa
from sklearn.preprocessing import normalize
import config as cfg
### end of data_all.py imports
### functions of data_all.py
def extract_logmel(y, sr, size):

    # normalization
    y = y.astype(np.float32)
    normalization_factor = 1 / np.max(np.abs(y))
    y = y * normalization_factor

    # random crop
    if len(y) <= size * sr:
        new_y = np.zeros((int(size * sr)+1, ))
        new_y[:len(y)] = y
        y = new_y

    # extract log mel spectrogram #####
    melspectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=1024, n_mels=cfg.N_MEL)
    logmelspec = librosa.power_to_db(melspectrogram)

    return logmelspec.T


def get_wave_norm(file):
    y, sr = librosa.load(file, sr=cfg.SR)
    
    ####### this +0.3 from 0.51 -> 0.54
    # add trim for comparison
    #y_trimmed, idx = librosa.effects.trim(y)
    y_trimmed = y.copy()
    # add hpss for comparison, use harmonic (h)
    h,p = librosa.effects.hpss(y_trimmed)
    ####### great code
    
    ## more experiment below: this doesn't improve a lot, instead it goes from 0.535 back to 0.49, and there exist file test_210.wav empty error, solved manually by replacing this file with some other 1 file. Also, this may work but you may add in extra time difference information and also take in to account: examine each file processed result, also, experiment more on this, e.g. .2 seconds or something else.
    # split using librosa, using harmonic component
    
    '''
    yhs = librosa.effects.split(h,top_db=30,hop_length=64)
    select = np.diff(yhs/sr)>.15
    select_audio = np.array([],dtype=h.dtype)
    for i in range(select.shape[0]):
        if select[i][0]:
            temp_y = h[yhs[i][0]:yhs[i][1]]
            new = np.concatenate([select_audio,temp_y])
            select_audio = new


    data = select_audio
    '''
    return h, sr




# SAVE VAL IN THE TEST WAY
if True:
    if True:#not os.path.isfile('data_test.pkl'):
        DATA_DIR = './input/val'

        file_glob = []
        for i, cls_fold in tqdm(enumerate(cfg.LABELS)):

            cls_base = os.path.join(DATA_DIR, cls_fold)
            files = os.listdir(cls_base)
            print('{} train num:'.format(cls_fold), len(files))
            for pt in files:
                file_pt = os.path.join(cls_base, pt)
                file_glob.append((file_pt, cfg.LABELS.index(cls_fold)))

        '''
        for cls_fold in tqdm(os.listdir(DATA_DIR)):

            file_pt = os.path.join(DATA_DIR, cls_fold)
            file_glob.append(file_pt)

        print(len(file_glob))
        '''
        print('done.')

        data = {}

        for file in tqdm(file_glob):

            temp = []

            raw, sr = get_wave_norm(file[0])
            length = raw.shape[0]
            seg = int(sr * cfg.TIME_SEG)
            for i in range((length//seg)*cfg.STRIDE+1):
                start = i * int(seg/cfg.STRIDE)
                end = start + seg
                if end <= length:
                    x = raw[start:end]
                    x = extract_logmel(x, sr, size=cfg.TIME_SEG)
                    temp.append(x)
            data[file] = np.array(temp)

        with open('VAL_TEST_WAY.pkl', 'wb') as f:
            pkl.dump(data, f)

'''
####### SAVE VAL X

if True:#not os.path.isfile('data_test.pkl'):
        DATA_DIR = './input/val'

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
            seg = int(sr * cfg.TIME_SEG)
            for i in range((length//seg)*cfg.STRIDE+1):
                start = i * int(seg/cfg.STRIDE)
                end = start + seg
                if end <= length:
                    x = raw[start:end]
                    x = extract_logmel(x, sr, size=cfg.TIME_SEG)
                    temp.append(x)
            data[file] = np.array(temp)

        with open('VAL_X.pkl', 'wb') as f:
            pkl.dump(data, f)


### SAVE VAL Y


    if True:#not os.path.isfile('data_test.pkl'):
        DATA_DIR = './input/val'

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
            seg = int(sr * cfg.TIME_SEG)
            for i in range((length//seg)*cfg.STRIDE+1):
                start = i * int(seg/cfg.STRIDE)
                end = start + seg
                if end <= length:
                    x = raw[start:end]
                    x = extract_logmel(x, sr, size=cfg.TIME_SEG)
                    temp.append(x)
            data[file] = np.array(temp)

        with open('data_test.pkl', 'wb') as f:
            pkl.dump(data, f)
'''
