# MOVE NUM_SAMPLES .wav files of each category to the corresponding folder
from pathlib import Path
import librosa
import librosa.display as disp
import os
import shutil
import soundfile
from tqdm import tqdm

NUM_SAMPLES = 10

def get_files(path):
    fs = librosa.util.find_files(path)
    return fs
def gen_save(f):
    dirname,fname = os.path.split(f)
    y, sr = librosa.load(f,sr=None)
    S = librosa.feature.melspectrogram(y=y,sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.figure()
    disp.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    save_fname = dirname + '/' + fname.split('.')[0] + '_mel.png'
    title = save_fname
    plt.title(title)
    plt.savefig(save_fname)
    plt.close()



# move TRAIN only:
cwd = Path.cwd()
train_dirs = os.listdir('input/train')
for d in train_dirs:
    class_dir = cwd/'input/train'/d
    print(class_dir)
    fs = get_files(class_dir)
    for f in fs[:NUM_SAMPLES]:
         dst = cwd/'input/eda/mel_samples'/d
         shutil.copy(f,dst)
         print('Copied ',f,'into',dst,'!')

## generate mels and save png
mel_samples_dirs = os.listdir('input/eda/mel_samples')
for d in mel_samples_dirs:
    class_dir = cwd/'input/eda/mel_samples'/d
    print(class_dir)
    fs = get_files(class_dir)
    for f in tqdm(fs):
        gen_save(f)

'''
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
    ### moving sr ==1600 may not be as good as expected!
    if sr == 1600:
        new_filename = f[:-4] + '_new' + '.wav'
        y,sr = librosa.load(f,sr=16000)
        soundfile.write(new_filename,y,sr)
        print('write new file: ',new_filename)
        shutil.move(new_filename,f)
        print('1600 to 16000------------')
        print('move from',new_filename,'to',f)

'''
