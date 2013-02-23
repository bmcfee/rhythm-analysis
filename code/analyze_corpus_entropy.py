#!/usr/bin/env python
'''
CREATED:2013-02-23 14:07:03 by Brian McFee <brm2132@columbia.edu>

Usage:

./analyze_corpus_entropy.py /path/to/files/*.mp3 /path/to/output
'''


import librosa, scipy.stats, numpy
import sys, os, glob
import multiprocessing as mp
import cPickle as pickle

# <codecell>

CORES       = 24
PARAMETERS = {
#     'TARGET_SR'    :  22050,
    'TARGET_SR'    :  8000,
    'FFT_WINDOW'   :  2048,
    'HOP_SIZE'     :  256,
    'MEL_BINS'     :  64,
    'BEAT_WINDOW'  :  128,
    'BEAT_CLEAR'   :  16
}
    
def audio_onset_correlation(infile):
    
    # Load the audio
    (y, sr) = librosa.load(infile, target_sr=PARAMETERS['TARGET_SR'])
    
    # Generate a mel spectrogram
    S = librosa.melspectrogram(y, sr, window_length=PARAMETERS['FFT_WINDOW'], hop_length=PARAMETERS['HOP_SIZE'], mel_channels=PARAMETERS['MEL_BINS'])
    
    # Generate per-band onsets
    onsets = numpy.empty( (S.shape[0], S.shape[1]-1) )
    
    for i in range(PARAMETERS['MEL_BINS']):
        onsets[i,:] = librosa.beat.onset_strength(y, sr, window_length=PARAMETERS['FFT_WINDOW'], hop_length=PARAMETERS['HOP_SIZE'], S=S[i:(i+1),:])
        pass
    
    # Per-band onset correlation
    P = 0.0

    for t in xrange(0, onsets.shape[1] - PARAMETERS['BEAT_WINDOW'] - PARAMETERS['BEAT_CLEAR']):
        P = P + numpy.dot(numpy.diag(onsets[:, t]), onsets[:, (t+PARAMETERS['BEAT_CLEAR']):(t+PARAMETERS['BEAT_WINDOW']+PARAMETERS['BEAT_CLEAR'])])
        pass
    
    return P 

def entropy_profile(P):

    z       = 1.0 / numpy.sum(P, axis=-1)
    Pnorm   = numpy.dot(numpy.diag(z), P)
    H       = scipy.stats.entropy(P.T) / numpy.log(P.shape[-1])

    return H

def save_output(outdir, filename, A, H):
    with open('%s/%s.pickle' % (outdir, os.path.basename(filename)), 'w') as f:
        pickle.dump({'filename': filename, 'A': A, 'H': H, 'P': PARAMETERS}, f, protocol=-1)
        pass
    pass

def analyze_files(inpath, outdir):
    files = glob.glob(inpath)
    files.sort()
    # TODO:   2013-02-23 14:09:07 by Brian McFee <brm2132@columbia.edu>
    #   parallelize this

    def __consumer(in_Q, out_Q):
        while True:
            try:
                filename    = in_Q.get(True, 1)
                analysis    = audio_onset_correlation(filename)
                entropy     = entropy_profile(analysis)
                save_output(outdir, filename, analysis, entropy)
                out_Q.put(filename)
            except:
                break
        out_Q.close()
        return

    in_Q    = mp.Queue()
    out_Q   = mp.Queue()

    for (i, filename) in enumerate(files):
        in_Q.put(filename)
        if i > 16:
            break
        pass

    for i in range(CORES):
        mp.Process(target=__consumer, args=(in_Q, out_Q)).start()
        pass

    for j in xrange(len(files)):
        out_Q.get(True)
        pass
    pass

if __name__ == '__main__':
    analyze_files(sys.argv[1], sys.argv[2])
    pass
