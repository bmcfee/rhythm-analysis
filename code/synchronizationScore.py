# synchronizationScore.py
# Calculate the synchronization between two perforumers based on their ODF
#
# Created by Colin Raffel on 11/12/12

import numpy as np
import scipy.interpolate

def padOrTruncate( array, size ):
  if size < 1:
    size = 1
  if array.shape[0] > size:
    return array[:size]
  else:
    return np.append( array, np.zeros( size - array.shape[0] ) )

def getScore( performer1ODF, performer2ODF, **kwargs ):
  offset = kwargs.get( 'offset', 20 )
  # Make it so that one score is 10 samples smaller than the other
  if performer1ODF.shape[0] > performer2ODF.shape[0]:
    performer2ODF = padOrTruncate( performer2ODF, performer1ODF.shape[0] - offset )
    smallerSize = performer2ODF.shape[0]
  else:
    performer1ODF = padOrTruncate( performer1ODF, performer2ODF.shape[0] - offset )
    smallerSize = performer1ODF.shape[0]
  # Get the correlation
  correlation = np.correlate( performer1ODF, performer2ODF, 'valid' )
  '''import matplotlib.pyplot as plt
  plt.plot( performer1ODF )
  plt.plot( performer2ODF )
  plt.show()'''
  # Return the max correlation, divided by the number of terms summed in it
  return np.max( correlation )/(1.0*smallerSize)