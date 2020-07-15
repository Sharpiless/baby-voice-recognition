# MOVE 1600 Hz sampling rate .wav files to the corresponding folder
from pathlib import Path
import librosa
import os
import shutil
import numpy as np

def get_files(path):
    fs = librosa.util.find_files(path)
    return fs


# move TRAIN only:
cwd = Path.cwd()
train_dirs = os.listdir('input/train')
for d in train_dirs:
    class_dir = cwd/'input/train'/d
    print(class_dir)
    fs = get_files(class_dir)
    # randomly choose 10 .wav files
    chosen = np.random.choice(fs,10,replace=False)
    dst = cwd/'input/val'/d
    for f in chosen:
        shutil.move(f,dst)
        #print('Moved ',f,'into',dst,'!')
    print('Finish moving 10 files of class',d,'!')

