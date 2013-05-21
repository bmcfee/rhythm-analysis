#!/usr/bin/env python
"""
Construct onset detection data

CREATED:2013-04-01 17:04:22 by Brian McFee <brm2132@columbia.edu>
"""

import sys
import os
import glob
import cPickle as pickle

import numpy as np

import librosa

def audio_to_examples(wavfile, onsetfile, tol=0.02, 
                      target_sr=22050, n_fft=512, hop_length=128):
    """Extract audio features from an input file.

    Features per frame:
        - magnitude spectrum    (+delta, +delta^2)
        - power spectrum        (+delta, +delta^2)
        - log-mag spectrum      (+delta, +delta^2)
        - phase                 (delta, delta^2) (unwrapped, pincarged)

    Arguments:
        wavfile     --  (string)    path to audio file
        onsetfile   --  (string)    path to file containing onset times
        tol         --  (float>0)   tolerance (in seconds) of onset times
        target_sr   --  (int>0)     target sample rate
        n_fft       --  (int>0)     FFT size
        hop_length  --  (int>0)     hop length

    Returns (X, Y):
        X           --  (ndarray)   n-by-d feature matrix. One row per example
        Y           --  (ndarray, bool) label vector corresponding to onsets
    """

    def get_mag_features(S):
        """Get magnitude spectrum features + delta, delta^2"""

        magspec     = np.abs(S)
        powspec     = magspec**2.0
        logmagspec  = librosa.logamplitude(S)

        X0 = np.vstack( (magspec, powspec, logmagspec) )
        X1 = np.diff(X0, axis=1, n=1)
        X2 = np.diff(X0, axis=1, n=2)

        return np.vstack( (X0[:, 2:], X1[:, 1:], X2) )

    def get_phase_features(S):
        """Get phase features: delta, delta^2"""

        P   = np.unwrap(np.angle(S))
        
        P1  = np.mod(np.diff(P, axis=1, n=1), np.pi)
        P2  = np.mod(np.diff(P, axis=1, n=2), np.pi)

        return np.vstack( (P1[:, 1:], P2) )

    def get_features():
        """Get all audio features"""
        (x, sr) = librosa.load(wavfile, sr=target_sr)

        S       = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)

        return np.vstack( (get_mag_features(S), get_phase_features(S)) )
    
    def time_to_frames(times):
        """Convert times to frame numbers"""
        return (times * float(target_sr) / float(hop_length)).astype(int)

    # 1. Get the features from the audio
    X = get_features()
    n = X.shape[1]

    # 2. load the onset times
    onset_times = np.loadtxt(onsetfile)

    # 3. fuzzy-match times to onsets to determine labels
    Y = np.zeros(n, dtype=bool)
    min_times = np.maximum(0, time_to_frames(onset_times - tol))
    max_times = np.minimum(n, time_to_frames(onset_times + tol))

    for (lb, up) in zip(min_times, max_times):
        Y[lb:up] = True
        pass

    return (X.T.astype(np.float16), Y)


def get_data(input_directory):

    wavs = glob.glob('%s/*.wav' % input_directory)

    wavs.sort()

    for wav in wavs:
        label = '%s.txt' % (os.path.splitext(wav)[0])
        yield (wav, label)

def filter_data(X, Y, tau=1e-3):

    ind = (X[:,:257]**2).sum(axis=1) > tau

    return X[ind,:], Y[ind]


def processFiles(input_directory,  output_pickle):
    """Crunches data into features

    Arguments:
        input_directory     --  (string)    path to directory containing data
        output_pickle       --  (string)    path to store feature/label data
    """

    X = None
    Y = None

    for (wav, onset) in get_data(input_directory):
        
        print os.path.basename(wav)
        (_X, _Y) = audio_to_examples(wav, onset)

        if X is None:
            X = _X
            Y = _Y
        else:
            X = np.vstack( (X, _X) )
            Y = np.hstack( (Y, _Y) )

        pass

    (X, Y) = filter_data(X, Y)

    with open(output_pickle, 'w') as outfile:
        pickle.dump({'X': X, 'Y': Y}, outfile, protocol=-1)


if __name__ == '__main__':
    indir       = sys.argv[1]
    outpickle   = sys.argv[2]

    processFiles(sys.argv[1], sys.argv[2])
