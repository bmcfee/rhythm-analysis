# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import librosa
import scipy.signal, scipy.stats
import numpy
import os

# <codecell>

def makeSpec(filename, scale=1.0, hop_size=64):
    (y, sr) = librosa.load(filename, sr=22050)
    S = librosa.feature.melspectrogram(y, sr, n_fft=2048, hop_length=hop_size, n_mels=128, fmax=8000) ** scale
    return (S, sr)

# <codecell>

def normalize(x):
    if x.max() == x.min():
        return x
    
    x = x - x.min()
    x = x / x.max()
    return x

# <codecell>

def onsetStrength(S):
    # Compute the first difference in time
    onset = numpy.diff(S, n=1, axis=1)
    
    # Discard negatives to catch rising edges
    onset = numpy.maximum(onset, 0.0)
    
    # Average over mel bands
    onset = numpy.mean(onset, axis=0)
    
    return normalize(onset)

# <codecell>

def multiband(S, band_size=16, hop_size=8):
    (d, t) = S.shape
    X = numpy.empty( ((d / hop_size) - 1, t-1))
    for (i, b) in enumerate(range(0, d - hop_size, hop_size)):
        X[i] = onsetStrength(S[b:(b+band_size),:])
        pass
    return X

# <codecell>

def onset_correlate(ref, X, window_size=256):
    
    (D,T) = X.shape
    ref = scipy.stats.zscore(ref)
    X   = scipy.stats.zscore(X, axis=1)
    H = 0.0
    
    for t in xrange(window_size, T-window_size):
        H = H + ref[t] * X[:, (t-window_size):(t+window_size)]
        pass
    
    # Normalize by number of sample frames
    H = H / (T - 2 * window_size)
    
    # Normalize each row by peak correlation
    z = 1.0 / numpy.max(numpy.abs(H), axis=1)
    H = numpy.dot(numpy.diag(z), H)
    return H

# <codecell>

def mboc(songname, W=256, hop=256, N=1, smooth=False):
        
    (S, sr) = makeSpec(songname, hop_size=hop)
    
    if smooth:
        S = scipy.signal.medfilt2d(S, kernel_size=(1,3))
        pass
    
    log_S = librosa.logamplitude(S)
    
    M = multiband(log_S, band_size=2, hop_size=1)
    
    # high-pass the reference
    #R = onsetStrength(S[-28:,:])
    
    # low-pass the reference
    #R = onsetStrength(S[:8,:])
    
    # HPSS reference
    #(H, P) = hpss(S, alpha=0.5, max_iter=50)
    #R = onsetStrength(P)
    
    # Full spectrum reference
    R = onsetStrength(log_S)
    
    # How many frames do we have total?
    T = len(R)
    
    # What's the width of analysis frames given that we want N displays?
    sample_width = T / N
    
    # Reverse the mel bins to frequencies
    #F = librosa.mel_to_hz(numpy.arange(0, S.shape[0] + 2) * (sr / 2.0) / (S.shape[0] + 1.0))

    for i in range(N):
        
        s = i * sample_width
        
        H = onset_correlate(R[s:(s + sample_width)], M[:,s:(s+sample_width)], window_size=W)
        
        #subplot(numpy.ceil(N/2.0), min(2, N), i+1)
        figure()
        I = imshow(H, aspect='auto', origin='lower', vmin=-1.0, vmax=1.0)
        #I.set_clim(-1.0,1.0)
        #if N <= 1:
        if True:
            xlab = numpy.arange(-W, W+32, 32)
            xticks(W + xlab, ['%d' % int(z * hop * 1000 / sr) for z in xlab], rotation=-90)
            xlabel('Lag (ms)')
            nmels    = H.shape[0]
            binfreqs = librosa.feature.mel_frequencies(n_mels=nmels, fmax=8000)
            yticks(range(0, nmels+1, nmels/6), map(lambda x: '%dHz' % int(x), binfreqs[1:-1:(nmels/6)]))
            #ylab = numpy.arange(0, S.shape[0], 16)
            #yticks(ylab, F[::16])
            #ylabel('Hz')
        else:
            axis('off')
            pass
        t1 = s * hop / sr
        t2 = (s + sample_width) * hop / sr
        time_start = '%02d:%02d' % (t1 / 60, t1 % 60)
        time_stop  = '%02d:%02d' % (t2 / 60, t2 % 60)
        #if N >= 1:
        #    title('[%s - %s]' % (time_start, time_stop))
        #if i == 1:
        if True:
            title('[%s - %s] %s' % (time_start, time_stop, os.path.basename(songname)))
            pass
        colorbar()
        pass
    pass

# <codecell>

def onset_correlate_online(ref, X, window_size=256, delay=6, alpha=0.5):
    (D, T) = X.shape
    
    H = numpy.zeros((D, 2 * window_size))
    
    for t in xrange(0, window_size, delay):
        yield H
        
    for t in xrange(window_size, T - window_size):
        newframe = ref[t] * X[:, (t-window_size):(t+window_size)]
        H = (1 - alpha) * H + alpha * newframe
        #z = numpy.max(numpy.abs(H), axis=1)**-1.0
        #z[numpy.isnan(z)] = 1.0
        #H = numpy.dot(numpy.diag(z), H)
        if t % delay == 0:
            z = numpy.max(numpy.abs(H), axis=1)**-1
            yield numpy.dot(numpy.diag(z), H)
            #yield H
            pass
        pass
    
    for t in xrange(0, window_size, delay):
        H = (1 - alpha) * H
        yield H
        
    pass

# <codecell>

import time, sys, os, pipes
from IPython.display import HTML

def onset_correlate_animate(audio_file, ref, X, window_size, sr=11025, hop_size=64, fps=30):
    
    # Figure out the delay in terms of audio samples
    
    delay = int(float(sr) / (fps * hop_size))
    
    # generate the video
    
    print 'Generating video frames...'
       
    figure()
    tempfiles = []
    xlab = numpy.arange(-window_size, window_size+32, 32)
    for (i, frame) in enumerate(onset_correlate_online(ref, X, window_size=window_size, delay=delay, alpha=0.005)):
        fname = '/tmp/_ocv_frame_%06d.png' % i
        imsave(fname=fname, arr=frame, format='png', origin='lower')
        tempfiles.append(fname)
        pass
    
    num_frames = len(tempfiles)
    len_song   = hop_size * len(ref) / float(sr)
    
    MY_FPS        = num_frames / len_song
    
    print 'Encoding video...'
    #mencoder mf:///tmp/_ocv_frame_*.png -mf type=png:fps=8 
    #-ovc lavc -lavcopts vcodec=wmv2  -oac mp3lame -audiofile Desktop/daft_punk-da_funk.wav -o tmp.avi
    os.system("/usr/bin/mencoder mf:///tmp/_ocv_frame_*.png -mf type=png:fps=%.2f -ovc lavc -lavcopts vcodec=wmv2 -oac mp3lame -audiofile %s -o /tmp/ocv_video.avi" % (MY_FPS, pipes.quote(audio_file)))
    
    for fname in tempfiles:
        os.remove(fname)
        pass
    
    print 'Embedding video?'
    video = open('/tmp/ocv_video.avi', 'rb').read()
    video_tag = '<video controls alt="test" src="data:video/x-avi;base64,{0}">'.format(video.encode('base64'))
    HTML(data=video_tag)
    
    pass
    

# <codecell>

def ocv(songname, W=128, hop=64):
    (S, sr) = makeSpec(songname, hop_size=hop)
    M = multiband(S, band_size=4, hop_size=1)
    R = onsetStrength(S)
    onset_correlate_animate(songname, R, M, window_size=W, sr=sr, hop_size=hop, fps=30)
    pass

# <codecell>

ocv('/home/bmcfee/data/CAL500/wav/daft_punk-da_funk.wav', W=384)

# <codecell>

ocv('/home/bmcfee/Desktop/data/rhythm/06 Hot House.mp3', W=256)

# <codecell>

ocv('/home/bmcfee/Desktop/data/rhythm/04 The Big Push.wav', W=256)

# <codecell>

ocv('/home/bmcfee/Desktop/04 Speak No Evil.wav', W=256)

# <codecell>

ocv('/home/bmcfee/Desktop/Jones-Resolution.wav', W=512)

# <codecell>

ocv('/home/bmcfee/Desktop/Williams-Pinocchio.wav', W=512)

# <codecell>

mboc('/home/bmcfee/data/CAL500/wav/daft_punk-da_funk.wav', W=128)
#mboc('/home/bmcfee/data/CAL500/wav/daft_punk-da_funk.wav', W=64, smooth=True)

# <codecell>

#mboc('/home/bmcfee/data/CAL500/wav/art_tatum-willow_weep_for_me.wav', W=128)
#mboc('/home/bmcfee/data/Autumn Leaves/14 Duke Ellington & His Orchestra - Autumn Leaves.wav', W=128)
#mboc('/home/bmcfee/git/jazzir/data/AutLeavesJamal.wav', W=128)
mboc('/home/bmcfee/data/jazz_cal10k/Miles Davis Quintet - Autumn Leaves.mp3', W=128)

# <codecell>

mboc('/home/bmcfee/Desktop/data/rhythm/04 The Big Push.mp3', W=128)

# <codecell>

mboc('/home/bmcfee/Desktop/data/rhythm/04 Speak No Evil.mp3', W=128)

# <codecell>

mboc('/home/bmcfee/Desktop/data/rhythm/04 The Big Push.mp3', W=128, N=6)
#ocv('/home/bmcfee/Desktop/04 The Big Push.wav')

# <codecell>

mboc('/home/bmcfee/Desktop/data/rhythm/04 Speak No Evil.mp3', W=128, N=6)

