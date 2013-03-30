# evaluate.py
# A variety of evaluation techniques for determining a beat tracker's accuracy
#
# Created by Colin Raffel on 8/29/11
# Based on the methods described in
#     Matthew E. P. Davies,    Norberto Degara, and Mark D. Plumbley. 
#     "Evaluation Methods for Musical Audio Beat Tracking Algorithms", 
#     Queen Mary University of London Technical Report C4DM-TR-09-06
#     London, United Kingdom, 8 October 2009.
# See also the Beat Evaluation Toolbox, https://code.soundsoftware.ac.uk/projects/beat-evaluation/

import numpy as np
import sys

# Goto is wrong; AMLc and AMLt are always the same as CMLc and CMLt.

# Wrapper class for getting all evaluation metrics
class BeatTrackerEvaluator:
    # Initialize
    def __init__( self, annotatedBeats, generatedBeats, minBeatTime = 5.0, fMeasureThreshold = 0.07, \
                             cemgilSigma = 0.04, pScoreThreshold = 0.2, gotoThreshold = 0.2, \
                             gotoMu = 0.2, gotoSigma = 0.2, continuityPhaseThreshold = 0.175, \
                             continuityPeriodThreshold = 0.175, informationGainBins = 41 ):
        
        # "Explicit is better than implicit"
        self.annotatedBeats = annotatedBeats
        self.generatedBeats = generatedBeats
        self.minBeatTime = minBeatTime
        self.fMeasureThreshold = fMeasureThreshold
        self.cemgilSigma = cemgilSigma
        self.pScoreThreshold = pScoreThreshold
        self.gotoThreshold = gotoThreshold
        self.gotoMu = gotoMu
        self.gotoSigma = gotoSigma
        self.continuityPhaseThreshold = continuityPhaseThreshold
        self.continuityPeriodThreshold = continuityPeriodThreshold
        self.informationGainBins = informationGainBins

        # Initialize metrics
        #self.fMeasure, self.cemgil, self.continuityCemgil, self.pScore, self.goto, self.continuityCMLc, \
        #self.continuityCMLt, self.continuityAMLc, self.continuityAMLt, self.informationGain = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        self.metrics = np.zeros( 10 )
        self.metricNames = ['F-Meas', 'Cemgil', ' Goto ', 'PScore', ' CMLc ', ' CMLt ', ' AMLc ', ' AMLt ', 'InGain', 'CemCon']
        # Baseline scores - see http://colinraffel.com/wiki/beat_tracker_accuracy_expectations
        self.baseline = np.array([0.5578, 0.4038, 0.1760, 0.6341, 0.2829, 0.4200, 0.4794, 0.6768, 0.0426, 0.4745])
        
        # Calculate metrics
        self.calculateMetrics()
    
    # Wrapper function for populating metric variables
    def calculateMetrics( self ):
        # Make sure we were supplied with good beat annotations
        if len(self.annotatedBeats.shape) == 1 and \
             self.annotatedBeats[self.annotatedBeats > self.minBeatTime].shape[0] > 1 and \
             np.max(self.annotatedBeats) < 30000 and \
             len(self.generatedBeats.shape) == 1 and \
             self.generatedBeats[self.generatedBeats > self.minBeatTime].shape[0] > 1 and \
             np.max(self.generatedBeats) < 30000:
            # Make sure beats are sorted
            self.annotatedBeats = np.sort(self.annotatedBeats)
            self.generatedBeats = np.sort(self.generatedBeats)
            # Ignore beats up to minBeatTime
            self.annotatedBeats = self.annotatedBeats[self.annotatedBeats > self.minBeatTime]
            self.generatedBeats = self.generatedBeats[self.generatedBeats > self.minBeatTime]
            # Populate metrics
            self.getFMeasure()
            self.getCemgil()
            self.getGoto()
            self.getPScore()
            self.getContinuity()
            self.getInformationGain()
        else:
            print "Error - the beat arrays should be one-dimensional arrays of seconds!    Setting all metrics to 0."
            #self.fMeasure, self.cemgil, self.continuityCemgil, self.pScore, self.goto, self.continuityCMLc, \
            #self.continuityCMLt, self.continuityAMLc, self.continuityAMLt, self.informationGain = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            self.metrics = np.zeros( 10 )

    # Calculate the F-measure - (2*correct/(2*correct + false positive + false negative)
    def getFMeasure( self ):
        # Values for calculating F measure
        falsePositives = 0.0
        falseNegatives = 0.0
        truePositives = 0.0
        # Local copy of beat arrays
        annotatedBeats = self.annotatedBeats
        generatedBeats = self.generatedBeats
        for n in np.arange( self.annotatedBeats.shape[0] ):
            # Calculate window edges
            windowMin = annotatedBeats[n] - self.fMeasureThreshold
            windowMax = annotatedBeats[n] + self.fMeasureThreshold
            # Find the (indeces of the) beats in the window
            beatsInWindow = np.nonzero(np.logical_and((generatedBeats >= windowMin), (generatedBeats <= windowMax)))[0]
            # Remove beats in the window so that they are only counted once
            generatedBeats = np.delete(generatedBeats, beatsInWindow)
            # No beats found in window - add a false negative
            if beatsInWindow.shape[0] == 0:
                falseNegatives += 1.0
            # One or more beats in the window - add a hit and false positives for each spurious beat
            elif beatsInWindow.shape[0] >= 1:
                truePositives += 1.0
                falsePositives += ( beatsInWindow.shape[0] - 1 ) if ( beatsInWindow.shape[0] > 1 ) else 0
        # Add in all remaining beats to false positives
        falsePositives += falsePositives + generatedBeats.shape[0]
        # Calculate F-measure ensuring that we don't divide by 0
        if 2.0*truePositives + falsePositives + falseNegatives > 0:
            self.metrics[0] = 2.0*truePositives/(2.0*truePositives + falsePositives + falseNegatives)
        else:
            self.metrics[0] = 0
    
    # Return metric variations of the annotated beats (double tempo, half tempo, etc)
    def getAnnotatedBeatVariations( self ):
    # Create annotations at twice the metric level
        doubleAnnotatedBeats = np.interp( np.arange(0, self.annotatedBeats.shape[0]-.5, .5), \
                                                                            np.arange(0, self.annotatedBeats.shape[0]), self.annotatedBeats )
        # Create matrix of metrical variations of the annotations
        annotatedBeats = {}
        # "True" annotations
        annotatedBeats[0] = self.annotatedBeats
        # Off-beat
        annotatedBeats[1] = doubleAnnotatedBeats[1::2]
        # Double tempo
        annotatedBeats[2] = doubleAnnotatedBeats
        # Half-tempo odd beats
        annotatedBeats[3] = self.annotatedBeats[::2]
        # Half-tempo even beats
        annotatedBeats[4] = self.annotatedBeats[1::2]
        return annotatedBeats
    
    # Calculate Cemgil's accuracy, based on a gaussian error function near each annotated beat
    def getCemgil( self ):
        # Get off-beat, half-tempo, etc versions of the annotations
        annotatedBeatVariations = self.getAnnotatedBeatVariations()
        # Accuracies for each variation
        accuracies = np.zeros( len( annotatedBeatVariations ) )
        for n in np.arange( len( annotatedBeatVariations ) ):
            # Get this annotation variation
            annotatedBeats = annotatedBeatVariations[n]
            # Cycle through beats
            for m in np.arange( annotatedBeats.shape[0] ):
                # Find the generated beat with smallest error relative to the annotated beat
                beatDiff = np.min( np.abs( annotatedBeats[m] - self.generatedBeats ) )
                # Add in the error (calculated via a gaussian error function) into the accuracy
                accuracies[n] += np.exp( -(beatDiff*beatDiff)/(2.0*self.cemgilSigma*self.cemgilSigma) )
            # Normalize the accuracy
            accuracies[n] /= .5*(self.generatedBeats.shape[0] + annotatedBeats.shape[0])
        # Raw accuracy with non-varied annotations
        self.metrics[1] = accuracies[0]
        # Maximal accuracy across all variations
        self.metrics[9] = np.max( accuracies )
    
    # Calculate Goto's accuracy, which is binary 1 or 0 depending on some specific heuristic criteria
    def getGoto( self ):
        # Error for each beat
        beatError = np.ones( self.annotatedBeats.shape[0] )
        # Flag for whether the annotated and generated beats are paired
        paired = np.zeros( self.annotatedBeats.shape[0] )
        # Keep track of Goto's three criteria
        gotoCriteria = 0
        for n in np.arange(1, self.annotatedBeats.shape[0]-1):
            # Get previous inner-annotated-beat-interval
            previousInterval = 0.5*(self.annotatedBeats[n] - self.annotatedBeats[n-1])
            # Window start - in the middle of the current beat and the previous
            windowMin = self.annotatedBeats[n] - previousInterval
            # Next inter-annotated-beat-interval
            nextInterval = 0.5*(self.annotatedBeats[n+1] - self.annotatedBeats[n])
            # Window end - in the middle of the current beat and the next
            windowMax = self.annotatedBeats[n] + nextInterval
            # Get generated beats in the window
            beatsInWindow = np.logical_and((self.generatedBeats >= windowMin), (self.generatedBeats <= windowMax))
            # False negative/positive
            if beatsInWindow.sum() == 0 or beatsInWindow.sum() > 1:
                paired[n] = 0
                beatError[n] = 1
            else:
                # Single beat is paired!
                paired[n] = 1
                # Get offset of the generated beat and the annotated beat
                offset = self.generatedBeats[beatsInWindow] - self.annotatedBeats[n]
                # Scale by previous or next interval
                if offset < 0:
                    beatError[n] = offset/previousInterval
                else:
                    beatError[n] = offset/nextInterval
        # Get indices of incorrect beats
        correctBeats = np.nonzero( np.abs(beatError) > self.gotoThreshold )[0]
        # All beats are correct (first and last will be 0 so always correct)
        if correctBeats.shape[0] < 3:
            # Get the track of correct beats
            track = beatError[correctBeats[0] + 1:correctBeats[-1] - 1]
            gotoCriteria = 1
        else:
            # Get the track of maximal length
            trackLength = np.max( np.diff( correctBeats ) )
            trackStart = np.nonzero( np.diff( correctBeats ) == trackLength )[0][0]
            # Is the track length at least 25% of the song?
            if trackLength - 1 > .25*(self.annotatedBeats.shape[0]-2):
                gotoCriteria = 1
                track = beatError[correctBeats[trackStart]:correctBeats[trackStart+1]]
        # If we have a track
        if gotoCriteria:
            # Are mean and std of the track less than the required thresholds?
            if np.mean( track ) < self.gotoMu and np.std( track ) < self.gotoSigma:
                gotoCriteria = 3
        # If all criteria are met, score is 100%!
        if gotoCriteria == 3:
            self.metrics[2] = 1.0
        else:
            self.metrics[2] = 0.0
    
    # Get McKinney's P-score, based on the autocorrelation of the annotated and generated beats
    def getPScore( self ):
        # Quantize beats to 10ms
        fs = 1.0/(0.010)
        # Get the largest time index
        endPoint = np.int( np.ceil( np.max( [np.max( self.generatedBeats ), np.max( self.annotatedBeats )] ) ) )
        # Make impulse trains with impulses at beat locations
        annotationsTrain = np.zeros( endPoint*fs + 1)
        annotationsTrain[np.array( np.ceil( self.annotatedBeats*fs ), dtype=np.int )] = 1.0
        generatedTrain = np.zeros( endPoint*fs + 1 )
        generatedTrain[np.array( np.ceil( self.generatedBeats*fs ), dtype=np.int )] = 1.0
        # Window size to take the correlation over (defined as .2*median(inter-annotation-intervals))
        w = np.round( self.pScoreThreshold*np.median( np.diff( np.nonzero( annotationsTrain )[0] ) ) )
        # Get full correlation
        trainCorrelation = np.correlate( annotationsTrain, generatedTrain, 'full' )
        # Get the middle element - note we are rounding down on purpose here
        middleLag = trainCorrelation.shape[0]/2
        # Truncate to only valid lags (those corresponding to the window)
        trainCorrelation = trainCorrelation[middleLag - w:middleLag + w + 1] 
        assert trainCorrelation.shape[0] == 2*w + 1
        self.metrics[3] = np.sum( trainCorrelation )/np.max( [self.generatedBeats.shape[0], self.annotatedBeats.shape[0]] )
    
    # Get metrics based on how much of the generated beat sequence is continually correct
    def getContinuity( self ):
        # Get off-beat, half-tempo, etc versions of the annotations
        annotatedBeatVariations = self.getAnnotatedBeatVariations()
        # Accuracies for each variation
        continuousAccuracies = np.zeros( len( annotatedBeatVariations ) )
        totalAccuracies = np.zeros( len( annotatedBeatVariations ) )
        # Get accuracy for each variation
        for n in np.arange( len( annotatedBeatVariations ) ):
            annotatedBeats = annotatedBeatVariations[n]
            # Annotations that have been used
            usedAnnotations = np.zeros( np.max( [annotatedBeats.shape[0], self.generatedBeats.shape[0]] ) )
            # Whether or not we are continuous at any given point
            beatSuccesses = np.zeros( np.max( [annotatedBeats.shape[0], self.generatedBeats.shape[0]] ) )
            # Is this beat correct?
            beatSuccess = 0
            for m in np.arange( self.generatedBeats.shape[0] ):
                beatSuccess = 0
                # Get differences for this beat
                beatDifferences = np.abs( self.generatedBeats[m] - annotatedBeats )
                # Get nearest annotation index
                minDifference = np.min( beatDifferences )
                nearestAnnotation = np.nonzero( beatDifferences == minDifference )[0][0]
                # Have we already used this annotation?
                if usedAnnotations[nearestAnnotation] == 0:
                    # Is this the first beat or first annotation?    If so, look forward
                    if (m == 0 or nearestAnnotation == 0) and (m + 1 < self.generatedBeats.shape[0]):
                        # How far is the generated beat from the annotated beat, relative to the inter-annotation-interval?
                        phase = np.abs( minDifference/( annotatedBeats[nearestAnnotation + 1] - \
                                        annotatedBeats[nearestAnnotation] ) )
                        # How close is the inter-beat-interval to the inter-annotation-interval?
                        period = np.abs( 1 - ( self.generatedBeats[m + 1] - self.generatedBeats[m] ) \
                                         /( annotatedBeats[nearestAnnotation + 1] - annotatedBeats[nearestAnnotation] ) ) 
                        if phase < self.continuityPhaseThreshold and period < self.continuityPeriodThreshold:
                            # Set this annotation as used
                            usedAnnotations[nearestAnnotation] = 1
                            # This beat is matched
                            beatSuccess = 1
                    # This beat/annotation is not the first
                    else:
                        # How far is the generated beat from the annotated beat, relative to the inter-annotation-interval?
                        phase = np.abs( minDifference/( annotatedBeats[nearestAnnotation] - \
                                        annotatedBeats[nearestAnnotation - 1] ) )
                        # How close is the inter-beat-interval to the inter-annotation-interval?
                        period = np.abs( 1 - ( self.generatedBeats[m] - self.generatedBeats[m - 1] ) \
                                         /( annotatedBeats[nearestAnnotation] - annotatedBeats[nearestAnnotation - 1] ) ) 
                        if phase < self.continuityPhaseThreshold and period < self.continuityPeriodThreshold:
                            # Set this annotation as used
                            usedAnnotations[nearestAnnotation] = 1
                            # This beat is matched
                            beatSuccess = 1
                # Set whether this beat is matched or not
                beatSuccesses[m] = beatSuccess
            # Add 0s at the begnning and end so that we at least find the beginning/end of the generated beats
            beatSuccesses = np.append( np.append( 0, beatSuccesses ), 0 )
            # Where is the beat not a match?
            beatFailures = np.nonzero( beatSuccesses == 0 )[0]
            # Take out those zeros we added
            beatSuccesses    = beatSuccesses[1:-1]
            # Get the continuous accuracy as the longest string of successful beats
            continuousAccuracies[n] = (np.max( np.diff( beatFailures ) ) - 1)/(1.0*beatSuccesses.shape[0])
            # Get the total accuracy - all sequences
            totalAccuracies[n] = np.sum( beatSuccesses )/(1.0*beatSuccesses.shape[0])
        # Grab accuracy scores
        self.metrics[4] = continuousAccuracies[0]
        self.metrics[5] = totalAccuracies[0]
        self.metrics[6] = np.max( continuousAccuracies )
        self.metrics[7] = np.max( totalAccuracies )
    
    # Get the information gain - K-L divergence of the beat error histogram to a uniform histogram
    def getInformationGain( self ):
        # Get entropy for annotated beats->generated beats and generated beats->annotated beats
        forwardEntropy = self.getEntropy( self.annotatedBeats, self.generatedBeats )
        backwardEntropy = self.getEntropy( self.generatedBeats, self.annotatedBeats )
        # Pick the larger of the entropies
        if forwardEntropy > backwardEntropy:
            # Note that the beat evaluation toolbox does not normalize
            self.metrics[8] = ( np.log2( self.informationGainBins ) - forwardEntropy )/self.informationGainBins
        else:
            self.metrics[8] = ( np.log2( self.informationGainBins ) - backwardEntropy )/self.informationGainBins
        
    # Helper function for information gain (needs to be run twice - once backwards, once forwards)
    def getEntropy( self, annotatedBeats, generatedBeats ):
        beatError = np.zeros( generatedBeats.shape[0] )
        for n in np.arange( generatedBeats.shape[0] ):
            # Get index of closest annotation to this beat
            beatDistances = generatedBeats[n] - annotatedBeats
            closestBeat = np.nonzero( np.abs( beatDistances ) == np.min( np.abs( beatDistances ) ) )[0][0]
            absoluteError = beatDistances[ closestBeat ]
            # If the first annotation is closest...
            if closestBeat == 0:
                # Inter-annotation interval - space between first two beats
                interval = .5*( annotatedBeats[1] - annotatedBeats[0] )
            # If last annotation is closest...
            if closestBeat == (annotatedBeats.shape[0] - 1):
                interval = .5*( annotatedBeats[-1] - annotatedBeats[-2] )
            else:
                if absoluteError > 0:
                    # Closest annotation is the one before the current beat - so look at previous inner-annotation-interval
                    interval = .5*( annotatedBeats[closestBeat] - annotatedBeats[closestBeat - 1] )
                else:
                    # Closest annotation is the one after the current beat - so look at next inner-annotation-interval
                    interval = .5*( annotatedBeats[closestBeat + 1] - annotatedBeats[closestBeat] )
            # The actual error of this beat
            beatError[n] = .5*absoluteError/interval
        # Trick to deal with bin boundaries
        beatError = np.round( 10000*beatError )/10000.0
        # Put beat errors in range (-.5, .5)
        beatError = np.mod( beatError + .5, -1 ) + .5
        # Bins for the histogram
        #binStep = 1.0/( self.informationGainBins - 2.0 )
        #histogramBins = np.append( -.5, np.append( np.arange( -.5 + .5*binStep, .5 + .5*binStep, binStep ), .5 ) )
        # Note these are slightly different than those used in the beat evaluation toolbox (they are uniform)
        binStep = 1.0/( self.informationGainBins - 1.0 )
        histogramBins = np.arange( -.5, .5 + binStep, binStep )
        # Get the histogram
        rawBinValues = np.histogram( beatError, histogramBins )[0]
        # Add the last bin height to the first bin
        rawBinValues[0] += rawBinValues[-1]
        # Turn into a proper probability distribution
        rawBinValues = rawBinValues/(1.0*np.sum( rawBinValues ))
        # Set zero-valued bins to 1 to make the entropy calculation well-behaved
        rawBinValues[rawBinValues == 0] = 1
        # Calculate entropy
        return -np.sum( rawBinValues * np.log2( rawBinValues ) )
     
    # Helper functions for printing out the metrics
    def printMetricNames( self ):
        return ' | '.join( self.metricNames )
    
    def metricsFormat( self ):
        stringFormatList = []
        for n in np.arange( self.metrics.shape[0] ):
            stringFormatList.append( '%1.4f' )
        formatString = ' | '.join( stringFormatList )
        return formatString
        
        
        
# Run function as script
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage: %s annotatedBeats.txt generatedBeats.txt" % sys.argv[0]
        sys.exit(-1)
    # Get beat arrays
    annotatedBeats = np.genfromtxt( sys.argv[1] )
    generatedBeats = np.genfromtxt( sys.argv[2] )
    # Run evaluator
    evaluator = BeatTrackerEvaluator( annotatedBeats, generatedBeats )
    # Print out results
    print evaluator.printMetricNames()
    print evaluator.metricsFormat() % tuple( evaluator.metrics )