import numpy as np
import config as cfg

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
    y_trimmed, idx = librosa.effects.trim(y)
    # add hpss for comparison, use harmonic (h)
    h,p = librosa.effects.hpss(y_trimmed)
    ####### great code
    
    ## more experiment below: this doesn't improve a lot, instead it goes from 0.535 back to 0.49, and there exist file test_210.wav empty error, solved manually by replacing this file with some other 1 file. Also, this may work but you may add in extra time difference information and also take in to account: examine each file processed result, also, experiment more on this, e.g. .2 seconds or something else.
    # split using librosa, using harmonic component
    
    
    yhs = librosa.effects.split(y_trimmed,top_db=40,frame_length=1024*8,hop_length=1024*4)
    select = np.diff(yhs/sr)>.15
    select_audio = np.array([],dtype=h.dtype)
    for i in range(select.shape[0]):
        if select[i][0]:
            temp_y = h[yhs[i][0]:yhs[i][1]]
            new = np.concatenate([select_audio,temp_y])
            select_audio = new
    if len(select_audio) >= sr * cfg.TIME_SEG:
        data = select_audio
    elif len(select_audio) < sr * cfg.TIME_SEG:
        data = h
    
    return data, sr




### end of functions of data_all.py
### model.py imports
import keras.backend as K
from keras import regularizers
from keras import layers
from keras.models import Sequential
import keras
import os
import wave
import numpy as np
import pickle as pkl
from keras.layers import GaussianNoise
import config as cfg
import json

### end of model.py
### test.py imports
import keras.backend as K
from keras import regularizers
from keras import layers
from keras.models import Sequential
import keras
import os
import wave
import numpy as np
import pickle as pkl

from tqdm import tqdm
import pandas as pd

from keras.models import load_model
import config as cfg

### end of test.py imports

# for constants
start = 1.4#1.385#0.5#1.39
end = 1.5#1.390#10.5#1.41
increment = 0.1#0.005#0.005
for duration in np.arange(start,end,increment):
    cfg.TIME_SEG = duration
    ### data_all.py
    if True:#not os.path.isfile('data.pkl'):
        DATA_DIR = './input/train'

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
            seg = int(sr * cfg.TIME_SEG)
            length = raw.shape[0]
            for i in range((length//seg)*cfg.STRIDE+1):
                start = i * int(seg/cfg.STRIDE)#seg/cfg.STRIDE means "walk length = segment length/cfg.STRIDE"
                end = start + seg
                if end <= length:
                    x = raw[start:end]
                    y = np.zeros(cfg.N_CLASS)
                    y[lbl] = 1
                    x = extract_logmel(x, sr, size=cfg.TIME_SEG)
                    data.append((x, y))
        print(len(data))

        with open('data.pkl', 'wb') as f:
            pkl.dump(data, f)
    ### end of data_all.py
    ### data_test.py
    if True:#not os.path.isfile('data_test.pkl'):
        DATA_DIR = './input/test'

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
    ### end of data_test.py
    ### data_val.py
    if True:#not os.path.isfile('data_val.pkl'):
        DATA_DIR = './input/val'

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
            seg = int(sr * cfg.TIME_SEG)
            length = raw.shape[0]
            for i in range((length//seg)*cfg.STRIDE+1):
                start = i * int(seg/cfg.STRIDE)#seg/cfg.STRIDE means "walk length = segment length/cfg.STRIDE"
                end = start + seg
                if end <= length:
                    x = raw[start:end]
                    y = np.zeros(cfg.N_CLASS)
                    y[lbl] = 1
                    x = extract_logmel(x, sr, size=cfg.TIME_SEG)
                    data.append((x, y))
        print(len(data))

        with open('data_val.pkl', 'wb') as f:
            pkl.dump(data, f)
    ### end of data_val.py
    ### model.py
    with open('./data.pkl', 'rb') as f:
        raw_data = pkl.load(f)
    with open('./data_val.pkl', 'rb') as f:
        raw_data_val = pkl.load(f)

    raw_x = []
    raw_y = []

    raw_x_val = []
    raw_y_val = []

    for x, y in raw_data:
        raw_x.append(x)
        raw_y.append(y)
    for x, y in raw_data_val:
        raw_x_val.append(x)
        raw_y_val.append(y)


    np.random.seed(5)
    np.random.shuffle(raw_x)
    np.random.shuffle(raw_x_val)

    np.random.seed(5)
    np.random.shuffle(raw_y)
    np.random.shuffle(raw_y_val)

    print(len(raw_x), raw_x[0].shape)
    print(len(raw_x_val), raw_x_val[0].shape)

    train_x = np.array(raw_x)
    val_x = np.array(raw_x_val)
    train_y = np.array(raw_y)
    val_y = np.array(raw_y_val)

    print(train_x.shape)

    model = Sequential()
    model.add(layers.Conv1D(32*2, 3, input_shape=(train_x.shape[1], train_x.shape[2]),
                            kernel_regularizer=regularizers.l2(1e-7),
                            activity_regularizer=regularizers.l1(1e-7)))
    model.add(GaussianNoise(0.1))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv1D(32*2, 3, activation='elu',
                    kernel_regularizer=regularizers.l1_l2(1e-7)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool1D())
    model.add(GaussianNoise(0.1))
    model.add(layers.Dropout(0.5))
    model.add(layers.Bidirectional(layers.LSTM(32*4, dropout=0.5, return_sequences=True,
                    kernel_regularizer=regularizers.l1_l2(1e-7))))
    model.add(GaussianNoise(0.1))
    model.add(layers.Bidirectional(layers.LSTM(32*4, dropout=0.5, return_sequences=True,
                    kernel_regularizer=regularizers.l1_l2(1e-7))))
    model.add(layers.LSTM(32*2,
                    kernel_regularizer=regularizers.l1_l2(1e-7)))
    model.add(GaussianNoise(0.1))
    model.add(layers.Dense(16*2, activation='elu',
                    kernel_regularizer=regularizers.l1_l2(1e-7)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(cfg.N_CLASS, activation="softmax"))
    model.summary()

    adam = keras.optimizers.adam(2e-5)

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam, metrics=['accuracy'])

    # Train model on dataset
    batch_size = cfg.BATCH_SIZE
    steps = len(train_x) // batch_size

    # model.load_weights('./my_model.h5')

    history = model.fit(x=train_x, y=train_y, batch_size=batch_size,
              epochs=cfg.EPOCHES, validation_data=(val_x,val_y), shuffle=True)

    model.save('./my_model.h5')
    # may be used with "with open xxx"
    json.dump(history.history,open('output/fit_history_duration_{}.json'.format(duration),'w'))
    # Read data from file:
    # data = json.load( open('fit_history_duration_{}.json'.format(duration)))

    ### end of model.py

    ### test.py
    with open('./data_test.pkl', 'rb') as f:
        raw_data = pkl.load(f)

    #model = load_model('my_model.h5')

    result = {'id': [], 'label': []}

    for key, value in tqdm(raw_data.items()):

        x = np.array(value)
        y = model.predict(x)
        y = np.mean(y, axis=0)

        pred = cfg.LABELS[np.argmax(y)]

        result['id'].append(os.path.split(key)[-1])
        result['label'].append(pred)

    result = pd.DataFrame(result)
    result.to_csv('./submission.csv', index=False)
    ### end of test.py
