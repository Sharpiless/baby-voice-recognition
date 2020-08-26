     ...:         x = np.array(value)
     ...:         y = model.predict(x)
     ...:         tempam = np.argmax(y,axis=1)
     ...:         mode = scipy.stats.mode(tempam)[0][0]
     ...:         #y = np.mean(y, axis=0)
     ...: 
     ...:         #pred = cfg.LABELS[np.argmax(y)]
     ...:         pred = cfg.LABELS[mode]

