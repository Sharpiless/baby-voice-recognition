import warnings
warnings.filterwarnings('ignore')
import tqdm
import librosa
def get_tempo(f):
    y,sr = librosa.load(f,sr=None)
    onset_env = librosa.onset.onset_strength(y,sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env,sr=sr)
    return tempo


TEMPOS = []
for sub in 'hungry hug uncomfortable sleepy diaper awake'.split(' '):
    path = 'input/train/' + sub
    fs = librosa.util.find_files(path)
    tempos = []
    for f in tqdm.tqdm(fs):
        tempo = get_tempo(f)
        #print(tempo)
        tempos.append(tempo[0])
    TEMPOS.append(tempos)
