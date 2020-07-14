import keras.backend as K
from keras import regularizers
from keras import layers
from keras.models import Sequential
import keras
import os
import wave
import numpy as np
import pickle as pkl
from keras.layers import GaussianNoise
import config as cfg

if __name__ == '__main__':

    with open('./data.pkl', 'rb') as f:
        raw_data = pkl.load(f)

    raw_x = []
    raw_y = []

    for x, y in raw_data:
        raw_x.append(x)
        raw_y.append(y)

    np.random.seed(5)
    np.random.shuffle(raw_x)
    np.random.seed(5)
    np.random.shuffle(raw_y)

    print(len(raw_x), raw_x[0].shape)

    train_x = np.array(raw_x)
    train_y = np.array(raw_y)

    print(train_x.shape)


    model = Sequential()
    model.add(layers.Conv1D(32, 3, input_shape=(24, 60),
                    kernel_regularizer=regularizers.l2(1e-7),
                    activity_regularizer=regularizers.l1(1e-7)))
    model.add(GaussianNoise(0.1))
    model.add(layers.Bidirectional(layers.LSTM(
        64, dropout=0.5, return_sequences=True,
                    kernel_regularizer=regularizers.l2(1e-7),
                    activity_regularizer=regularizers.l1(1e-7))))
    model.add(layers.Bidirectional(layers.LSTM(
        64, dropout=0.5, return_sequences=True,
                    kernel_regularizer=regularizers.l2(1e-7),
                    activity_regularizer=regularizers.l1(1e-7))))
    model.add(layers.LSTM(64,
                    kernel_regularizer=regularizers.l2(1e-7),
                    activity_regularizer=regularizers.l1(1e-7)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(cfg.N_CLASS, activation="softmax"))
    model.summary()

    adam = keras.optimizers.adam(2e-4)

    model.compile(loss='categorical_crossentropy',
                optimizer=adam, metrics=['accuracy'])

    # Train model on dataset
    batch_size = cfg.BATCH_SIZE
    steps = len(train_x) // batch_size

    model.load_weights('./my_model.h5')

    model.fit(x=train_x, y=train_y, batch_size=batch_size,
            epochs=cfg.EPOCHES, validation_split=0.1, shuffle=True)

    model.save('./my_model.h5')
