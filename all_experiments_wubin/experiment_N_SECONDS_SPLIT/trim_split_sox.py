from tqdm import tqdm
import librosa
import soundfile
import os
import numpy as np
def load_trim_split_save(src_path,dst_path):
    fs = librosa.util.find_files(src_path)
    for f in tqdm(fs):
        y, sr = librosa.load(f,sr=None)
        y_trimmed, idx = librosa.effects.trim(y)
        y_splitted = librosa.effects.split(y_trimmed,top_db=30,hop_length=64)
        select_bool = np.diff(y_splitted/sr)>.2
        select_audio = np.array([],dtype=y.dtype)
        for i in range(select_bool.shape[0]):
            if select_bool[i][0]:
                temp_y = y_trimmed[y_splitted[i][0]:y_splitted[i][1]]
                new = np.concatenate([select_audio,temp_y])
                select_audio = new
        file_to_save = dst_path + '/' + f.split('/')[-1]
        soundfile.write(file_to_save,select_audio,sr)
        #librosa.output.write_wav(file_to_save,select_audio,sr)

for j in ['train','val']:
    src_path = 'input/' + j
    dst_path = 'input/eda/trimmed/' + j
    for i in ['hug', 'hungry', 'uncomfortable', 'awake', 'sleepy', 'diaper']:
        from_path = src_path + '/' + i
        to_path = dst_path + '/' + i
        print(from_path,to_path)
        load_trim_split_save(from_path,to_path)
for j in ['test']:
    src_path = 'input/' + j
    dst_path = 'input/eda/trimmed/' + j
    load_trim_split_save(src_path,dst_path)
    


######################
def sox_path(src_path,trimmed_path,concat_path):
    fs = librosa.util.find_files(src_path)
    for f in tqdm(fs):
        trimmed_wav_path = trimmed_path + '/' + f.split('/')[-1]
        concat_wav_path = concat_path + '/' + f.split('/')[-1]
        f1 = f
        f2 = 'bridge.wav'
        f3 = trimmed_wav_path
        f4 = concat_wav_path
        mycommand = 'sox {} {} {} {}'.format(f1,f2,f3,f4)
        os.system(mycommand)
    


for j in ['train','val']:
    src_path = 'input/' + j
    trimmed_path = 'input/eda/trimmed/' + j
    concat_path = 'input/eda/orig_concat_trim/' + j
    for i in ['hug', 'hungry', 'uncomfortable', 'awake', 'sleepy', 'diaper']:
        src_path = src_path + '/' + i
        trimmed_path = trimmed_path + '/' + i
        print(src_path,trimmed_path)
        concat_path = concat_path + '/' + i
        sox_path(src_path,trimmed_path,concat_path)
for j in ['test']:
    src_path = 'input/' + j
    trimmed_path = 'input/eda/trimmed/' + j
    concat_path = 'input/eda/orig_concat_trim/' + j 
    sox_path(src_path,trimmed_path,concat_path)



