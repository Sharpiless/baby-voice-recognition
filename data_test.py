import os
import wave
import numpy as np
from tqdm import tqdm
import pickle as pkl
from data_all import extract_logmel, get_wave_norm
import config as cfg

if __name__ == '__main__':
    
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
        seg = sr * cfg.TIME_SEG
        for i in range((length//seg)*cfg.STRIDE+1):
            start = i * int(seg/cfg.STRIDE)
            end = start + seg
            if end > length:
                end = length
            if end - start > sr * 1:
                x = raw[start:end]
            else:
                break
            x = extract_logmel(x, sr, size=cfg.TIME_SEG)
            temp.append(x)
        data[file] = np.array(temp)

    with open('data_test.pkl', 'wb') as f:
        pkl.dump(data, f)
