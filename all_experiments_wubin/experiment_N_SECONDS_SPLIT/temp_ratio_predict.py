    with open('./data_test.pkl', 'rb') as f:
        raw_data = pkl.load(f)

    #model = load_model('my_model.h5')

    result = {'id': [], 'label': []}

    for key, value in tqdm(raw_data.items()):

        x = np.array(value)
        y = model.predict(x)
        y = np.mean(y, axis=0)

        pred = cfg.LABELS[np.argmax(y)]

        result['id'].append(os.path.split(key)[-1])
        result['label'].append(pred)

    result = pd.DataFrame(result)
    result.to_csv('./submission.csv', index=False)


ymax = np.max(y,axis=1)
ymax2 = np.array([ymax,ymax,ymax,ymax,ymax,ymax]).T
ratio = ymax2/y
pred_idx = np.argmin(ratio.sum(axis=0))
pred = cfg.LABELS[pred_idx]


