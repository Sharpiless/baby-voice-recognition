import warnings
warnings.filterwarnings('ignore')
import tqdm
import librosa
def get_rms(f):
    y,sr = librosa.load(f,sr=None)
    rms = librosa.feature.rms(y)
    return rms


RMSS = []
for sub in 'hungry hug uncomfortable sleepy diaper awake'.split(' '):
    path = 'input/train/' + sub
    fs = librosa.util.find_files(path)
    rmss = []
    for f in tqdm.tqdm(fs):
        rms = get_rms(f)
        #print(tempo)
        rmss.append(rms[0])
    RMSS.append(rmss)

