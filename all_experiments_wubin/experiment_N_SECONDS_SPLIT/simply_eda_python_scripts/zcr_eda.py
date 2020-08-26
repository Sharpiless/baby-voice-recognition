import warnings
warnings.filterwarnings('ignore')
import tqdm
import librosa
def get_zcr(f):
    y,sr = librosa.load(f,sr=None)
    zcr = librosa.feature.zero_crossing_rate(y)
    return zcr


ZCRS = []
for sub in 'hungry hug uncomfortable sleepy diaper awake'.split(' '):
    path = 'input/train/' + sub
    fs = librosa.util.find_files(path)
    zcrs = []
    for f in tqdm.tqdm(fs):
        zcr = get_zcr(f)
        #print(tempo)
        zcrs.append(zcr[0])
    ZCRS.append(zcrs)
