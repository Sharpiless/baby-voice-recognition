# MOVE 1600 Hz sampling rate .wav files to the corresponding folder
from pathlib import Path
import librosa
import os
import shutil
import soundfile


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
    for f in fs:
        sr = librosa.get_samplerate(f)
        if sr == 1600:
            dst = cwd/'input/eda/1600'/d
            shutil.move(f,dst)
            print('Moved ',f,'into',dst,'!')
        if sr == 44100:
            new_filename = f[:-4] + '_new' + '.wav'
            y,sr = librosa.load(f,sr=16000)
            soundfile.write(new_filename,y,sr)
            print('write new file: ',new_filename)
            shutil.move(new_filename,f)
            print('44100 to 16000------------')
            print('move from',new_filename,'to',f)

# for test data:

print('---------------for test data----------------')
test_dir = cwd/'input/test'
fs = get_files(test_dir)
for f in fs:
    sr = librosa.get_samplerate(f)
    if sr == 44100:
        new_filename = f[:-4] + '_new' + '.wav'
        y,sr = librosa.load(f,sr=16000)
        soundfile.write(new_filename,y,sr)
        print('write new file: ',new_filename)
        shutil.move(new_filename,f)
        print('44100 to 16000------------')
        print('move from',new_filename,'to',f)

