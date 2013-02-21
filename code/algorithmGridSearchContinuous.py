# algorithmGridSearchContinuous.py
# Grid search for synchronization algorithm across many synchronizations
#
# Created by Colin Raffel on 12/3/12

import sys
import itertools
import numpy as np
import utility
import os
import csv
import onsetDetection
import synchronizationScore
import collections
import matplotlib.pyplot as plt
import time
import scipy.signal

if __name__ == "__main__":
  if len(sys.argv) < 3:
    print "Usage: %s datasetDirectory csvFileName.csv" % sys.argv[0]
    sys.exit(-1)
  
  ''' test
  ODFs = [onsetDetection.ODF.spectralDistance]
  downsamplingFactors = np.array([1])
  frameSizes = np.array([1024])
  hopSizeScales = np.array([8])
  windows = [np.hanning]
  offsets = np.array([20]) '''
  
  ''' For continuous '''
  ODFs = [onsetDetection.ODF.spectralDistance,\
          onsetDetection.ODF.melDifference,\
          onsetDetection.ODF.complex]
  downsamplingFactors = np.array([1])
  frameSizes = np.array([1024, 2048, 4096])
  hopSizeScales = np.array([8, 4])
  windows = [np.hanning]
  offsets = np.array([20])
    
  # Get subdirectories for the input folder, corresponding to different MIDI files
  directories = [os.path.join( sys.argv[1], folder ) for folder in os.listdir(sys.argv[1]) if os.path.isdir(os.path.join(sys.argv[1], folder)) and folder[0] is not '.']
  
  # The variations on the MIDI files
  filenames = []
  for shift in xrange( 0, 60, 10 ):
    filenames += ['0-' + str(shift) + 'ms.wav']
    filenames += ['1-' + str(shift) + 'ms.wav']
  
  nFiles = len( filenames )

  # Calculate number of tests about to be run
  nTests = np.product( [len(dimension) for dimension in (directories, ODFs, downsamplingFactors, frameSizes, hopSizeScales, windows, offsets)] )
  print "About to run " + str( nTests ) + " tests."
  # Keep track of which test is being run
  testNumber = 0
  
  startTime = time.time()
  
  # Store the parameters corresponding to each result
  gridSearchResults = collections.defaultdict(list)

  # The data, being manipulated each step of the way
  audioData = {}
  audioDataDownsampled = {}
  spectrograms = {}
  ODFOutput = {}
  synchronizationScores = {}
  
  # Test to plot histograms
  allAccuracies = np.zeros( nTests )
  
  # Can we calculate the spectrogram from the previous spectrogram?
  previousHopSizeScale = 0
  
  for directory in directories:
    # Read in wav data for each file - should try downsampling factors
    for file in filenames: audioData[file], fs = utility.getWavData( os.path.join( directory, file ) )
    for downsamplingFactor in downsamplingFactors:
      for file in filenames: audioDataDownsampled[file] = scipy.signal.decimate( audioData[file], downsamplingFactor )
      for frameSize, window, hopSizeScale in itertools.product( frameSizes, windows, hopSizeScales ):
        if hopSizeScale < previousHopSizeScale and np.mod( previousHopSizeScale, hopSizeScale ) == 0:
          # Instead of calculating a new spectrogram, just grab the frames
          newHopRatio = previousHopSizeScale/hopSizeScale
          for file in filenames: spectrograms[file] = spectrograms[file][::newHopRatio]
        else:
          # Calculate spectrograms - should not re-calculate if just the hop size changes.
          for file in filenames: spectrograms[file] = utility.getSpectrogram( audioDataDownsampled[file], hop=frameSize/hopSizeScale, frameSize=frameSize, window=window( frameSize ) )
        previousHopSizeScale = hopSizeScale
        for ODF in ODFs:
          # Get the onset detection function
          for file in filenames: ODFOutput[file] = onsetDetection.ODF( spectrograms[file], ODF, fs=fs/downsamplingFactor ).onsetDetectionFunction
          for offset in offsets:
            # Compute the synchronization score for the syncrhonized and unsynchronized files
            for n in xrange( nFiles/2 ):
              synchronizationScores[n] = synchronizationScore.getScore( ODFOutput[filenames[2*n]], ODFOutput[filenames[2*n + 1]], offset=offset )
            # Add in the ratio of the scores, we will take the per-MIDI-file-average later.
            print "{} -> {}, {:.3f}% done in {:.3f} minutes".format( (directory, ODF.__name__, downsamplingFactor, frameSize, hopSizeScale, window.__name__, offset), synchronizationScores.values(), (100.0*testNumber)/nTests, (time.time() - startTime)/60.0)
            testNumber += 1
            gridSearchResults[(ODF.__name__, downsamplingFactor, frameSize, hopSizeScale, window.__name__, offset)] += [np.array(synchronizationScores.values())]
  
  # Write out CSV results
  csvWriter = csv.writer( open( sys.argv[2], 'wb' ) )
  for parameters, results in gridSearchResults.items():
    resultArray = np.array( results )
    resultArray = (resultArray.T/np.max( resultArray, axis=1 )).T
    csvWriter.writerow( list( parameters ) + list(np.mean( resultArray, axis=0 )) + [np.mean( np.diff( resultArray ), axis=0 )] + [np.mean( np.diff( resultArray ) )] + [np.mean( np.sum( np.diff( resultArray ) < 0, axis=1 ) )/(0.5*nFiles - 1)] )