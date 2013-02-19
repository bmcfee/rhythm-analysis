# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import librosa
import os
import glob
import scipy.stats

# <codecell>

TARGET_SR    =  22050
FFT_WINDOW   =  2048
HOP_SIZE     =  256
MEL_BINS     =  64
BEAT_WINDOW  =  128
BEAT_CLEAR   =  16
    
def analyze_audio(infile):
    
    # Load the audio
    (y, sr) = librosa.load(infile, target_sr=TARGET_SR)
    
    
    # Generate a mel spectrogram
    S = librosa.melspectrogram(y, sr, window_length=FFT_WINDOW, hop_length=HOP_SIZE, mel_channels=MEL_BINS)
    
    # Generate per-band onsets
    onsets = numpy.empty( (S.shape[0]+1, S.shape[1]-1) )
    
    for i in range(MEL_BINS):
        onsets[i,:] = librosa.beat.onset_strength(y, sr, window_length=FFT_WINDOW, hop_length=HOP_SIZE, S=S[i:(i+1),:])
        pass
    
    # Generate the global onset profile
    onsets[-1,:] = librosa.beat.onset_strength(y, sr, window_length=FFT_WINDOW, hop_length=HOP_SIZE, S=S)
    
    # Per-band onset correlation
    P = 0.0

    for t in xrange(0, onsets.shape[1] - BEAT_WINDOW - BEAT_CLEAR):
        P = P + numpy.dot(numpy.diag(onsets[:, t]), onsets[:, (t+BEAT_CLEAR):(t+BEAT_WINDOW+BEAT_CLEAR)])
        pass
    
    return P 

def display_entropy(infile, P):
    
    z = 1.0 / numpy.sum(P, axis=1)
    Pnorm = numpy.dot(numpy.diag(z), P)
    
    figure()
    subplot(122)
    imshow(Pnorm[:-2,:], origin='lower', aspect='auto', interpolation='none', vmin=0)
    axis('tight')
    title('Sub-band onset autocorrelation')
    xlabel('Frame lag (ms)')
    
    xticks(
            range(0, BEAT_WINDOW + 1, BEAT_WINDOW / 4), 
            numpy.arange(BEAT_CLEAR, BEAT_CLEAR + BEAT_WINDOW + 1, BEAT_WINDOW / 4, dtype=int) * HOP_SIZE * 1000 / TARGET_SR)
    
    yticks([])
    colorbar(), draw()
    
    # compute entropy profile
    entropy = scipy.stats.entropy(P.T)
    
    # Normalize entropy by the uniform distribution (upper bound)
    entropy = entropy / numpy.log(P.shape[1])
    
    weights = 1.0 - entropy[:-1]
    weights = weights / numpy.sum(weights)
    pbar = numpy.dot(weights, P[:-1,:])
    hbar = scipy.stats.entropy(pbar) / numpy.log(P.shape[1])
    
    subplot(121)
    plot(entropy[:-1], range(len(entropy)-1))
    plot([entropy[-1], entropy[-1]], [0, len(entropy)-2], 'r--')
    plot([hbar, hbar], [0, len(entropy)-2], 'g--')
    axis([0, 1.0, 0, len(entropy)-2])
    
    legend(['Sub-band entropy', 'Global entropy', 'Weighted entropy'], loc='upper left')
    title('Onset correlation entropy')
    
    ylabel('Mel band')
    xlabel('H / Hmax')
    
    suptitle(os.path.basename(infile))
    pass

# <codecell>

def analyze_files(path):
    files = glob.glob(path)
    files.sort()
    for f in files:
        display_entropy(f, analyze_audio(f))
        pass
    pass

# <markdowncell>

# Note that since negentropy is convex, we have
# $$
#    H\left(\sum_i \mu_i p_i\right) \leq \sum_i \mu_i H\left(p_i\right)
# $$
# for distributions $p_i$ and $\mu$.

# <headingcell level=1>

# CAL500

# <codecell>

analyze_files('/home/bmcfee/data/CAL500/wav/charles_mingus-mood_indigo.wav')

# <codecell>

analyze_files('/home/bmcfee/data/CAL500/mp3/d*.mp3')

# <headingcell level=1>

# J-Disc

# <codecell>

analyze_files('/home/bmcfee/Desktop/data/rhythm/drums/[ABC]*')

# <codecell>

analyze_files('/home/bmcfee/Desktop/data/rhythm/drums/[DEF]*')

# <codecell>

analyze_files('/home/bmcfee/Desktop/data/rhythm/drums/[GHI]*')

