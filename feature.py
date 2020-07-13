import argparse
import concurrent.futures
import librosa
import numpy as np
import os


def track_features(path):
    print("Extracting ", path)

    y, sr = librosa.load(path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    # spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    # chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    # spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)

    features = mfcc.T
    return features

print(track_features('test/test_0.wav').shape)