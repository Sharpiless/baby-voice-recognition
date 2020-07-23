import json
import os
fs = os.listdir('output')
for f in fs:
    data = json.load(open('output/'+f))
    print(max(data['val_acc']),np.mean(data['val_acc']),f)

    

