# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.


import os
import sys
import json
import random
import argparse
import essentia
import essentia.streaming
from essentia.standard import *
import librosa
import numpy as np
from .extractor import FeatureExtractor
import pickle
from tqdm import tqdm
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--input_video_dir', type=str, default='/data1/diaoran/Data/I-Dancers/music')
parser.add_argument('--store_dir', type=str, default='/data1/diaoran/Data/I-Dancers/librosa')
parser.add_argument('--sampling_rate', type=int, default=15360)
args = parser.parse_args()
store_dir = args.store_dir
extractor = FeatureExtractor()

if not os.path.exists(args.store_dir):
    os.mkdir(args.store_dir)

def make_music_dance_set(video_dir, store_dir):
    print('---------- Extract features from raw audio ----------')
    musics = []
    dances = []
    fnames = []
    train = []
    test = []

    audio_fnames = sorted(os.listdir(video_dir))

    ii = 0
    # all_names = train + test
    for audio_fname in tqdm(audio_fnames):
        print(audio_fname)
        video_file = audio_fname

        sr = args.sampling_rate
        loader = essentia.standard.MonoLoader(filename=os.path.join(video_dir, video_file), sampleRate=sr)

        audio = loader()
        audio = np.array(audio).T

        feature = extract_acoustic_feature(audio, sr)
        print(feature.shape)

        filename = os.path.join(store_dir, audio_fname.replace('.mp3', '.pkl'))
        with open(filename, 'wb') as file:
            pickle.dump({'music': feature}, file)


def extract_acoustic_feature(audio, sr):

    melspe_db = extractor.get_melspectrogram(audio, sr)
    
    mfcc = extractor.get_mfcc(melspe_db)
    mfcc_delta = extractor.get_mfcc_delta(mfcc)
    # mfcc_delta2 = get_mfcc_delta2(mfcc)

    audio_harmonic, audio_percussive = extractor.get_hpss(audio)
    # harmonic_melspe_db = get_harmonic_melspe_db(audio_harmonic, sr)
    # percussive_melspe_db = get_percussive_melspe_db(audio_percussive, sr)
    chroma_cqt = extractor.get_chroma_cqt(audio_harmonic, sr,  octave=4)
    # chroma_stft = extractor.get_chroma_stft(audio_harmonic, sr)
    if chroma_cqt.shape[1] > mfcc.shape[1]:
        chroma_cqt = chroma_cqt[:, :-1]

    onset_env = extractor.get_onset_strength(audio_percussive, sr)
    tempogram = extractor.get_tempogram(onset_env, sr)
    onset_beat = extractor.get_onset_beat(onset_env, sr)[0]
    # onset_tempo, onset_beat = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    # onset_beats.append(onset_beat)

    onset_env = onset_env.reshape(1, -1)

    feature = np.concatenate([
        # melspe_db,
        mfcc, # 20
        mfcc_delta, # 20
        # mfcc_delta2,
        # harmonic_melspe_db,
        # percussive_melspe_db,
        # chroma_stft,
        chroma_cqt, # 12
        onset_env, # 1
        onset_beat, # 1
        tempogram
    ], axis=0)

            # mfcc, #20
            # mfcc_delta, #20

            # chroma_cqt, #12
            # onset_env, # 1
            # onset_beat, #1

    feature = feature.transpose(1, 0)
    print(f'acoustic feature -> {feature.shape}')

    return feature



if __name__ == '__main__':
    make_music_dance_set(args.input_video_dir) 
