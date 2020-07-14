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

if __name__ == '__main__':
    
    with open('./data_test.pkl', 'rb') as f:
        raw_data = pkl.load(f)

    model = load_model('my_model.h5')

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
