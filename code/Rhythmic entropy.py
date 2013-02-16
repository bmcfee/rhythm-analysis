# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import librosa
import os
import glob

# <codecell>

def analyze_audio(infile):
    TARGET_SR    =  22050
    FFT_WINDOW   =  2048
    HOP_SIZE     =  256
    MEL_BINS     =  64
    BEAT_WINDOW  =  64
    BEAT_CLEAR   =  8
    
    # Load the audio
    (y, sr) = librosa.load(infile, target_sr=TARGET_SR)
    
    
    # Generate a mel spectrogram
    S = librosa.melspectrogram(y, sr, window_length=FFT_WINDOW, hop_length=HOP_SIZE, mel_channels=MEL_BINS)
    
    # Generate per-band onsets
    onsets = numpy.empty( (S.shape[0], S.shape[1]-1) )
    
    for i in range(MEL_BINS):
        onsets[i,:] = librosa.beat.onset_strength(y, sr, window_length=FFT_WINDOW, hop_length=HOP_SIZE, S=S[i:(i+1),:])
        pass
    
    # Per-band onset correlation
    H = 0.0

    for t in xrange(0, onsets.shape[1] - BEAT_WINDOW - BEAT_CLEAR):
        H = H + numpy.dot(numpy.diag(onsets[:, t]), onsets[:, (t+BEAT_CLEAR):(t+BEAT_WINDOW+BEAT_CLEAR)])
        pass
    
    # Add a smoothing factor for the parts that never hit
    H = H + 1e-8
    
    z = numpy.sum(H, axis=1)
    H = numpy.dot(numpy.diag(1.0/z), H)
    
    figure()
    subplot(122), imshow(H, origin='lower', aspect='auto', interpolation='none'), axis('tight'), title('Sub-band onset autocorrelation')
    xlabel('Frame lag')
    xticks(range(0, BEAT_WINDOW, BEAT_WINDOW / 5), range(BEAT_CLEAR, BEAT_CLEAR + BEAT_WINDOW, BEAT_WINDOW / 5))
    yticks([])
    colorbar(), draw()
    
    # compute entropy profile
    z = H < 1e-10
    H[z] = 1.0
    entropy = - numpy.sum( H * numpy.log2(H), axis=1)
    
    # Normalize entropy by the uniform distribution (upper bound)
    entropy = entropy / numpy.log2(H.shape[1])
    
    subplot(121), plot(entropy, range(MEL_BINS)), axis([0, 1.0, 0, MEL_BINS-1]), title('Normalized sub-band entropy')
    ylabel('Mel band')
    xlabel('H')
    suptitle(os.path.basename(infile))
    pass

# <codecell>

def analyze_files(path):
    files = glob.glob(path)
    files.sort()
    for f in files:
        analyze_audio(f)
        pass
    pass

# <headingcell level=1>

# CAL500

# <codecell>

analyze_files('/home/bmcfee/data/CAL500/mp3/*.mp3')

# <headingcell level=1>

# J-Disc

# <codecell>

analyze_files('/home/bmcfee/Desktop/data/rhythm/drums/C*.m4a')

