# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# ## Calculates the synchronization between two perforumers based on their ODF

# <codecell>

import numpy as np
import scipy.stats

def padOrTruncate( array, size ):
    if size < 1:
        size = 1
    if array.shape[0] > size:
        return array[:size]
    else:
        return np.append( array, np.zeros( size - array.shape[0] ) )

def getScore( performer1ODF, performer2ODF, **kwargs ):
    offset = kwargs.get( 'offset', 20 )
    # Make it so that one score is "offset" samples smaller than the other
    if performer1ODF.shape[0] > performer2ODF.shape[0]:
        performer2ODF = padOrTruncate( performer2ODF, performer1ODF.shape[0] - offset )
        smallerSize = performer2ODF.shape[0]
    else:
        performer1ODF = padOrTruncate( performer1ODF, performer2ODF.shape[0] - offset )
        smallerSize = performer1ODF.shape[0]
    # Get the correlation
    correlation = np.correlate( performer1ODF, performer2ODF, 'valid' )
    # Make correlation sum to 1 so that it can be interpreted as a probability distribution
    correlation /= np.sum( correlation )
    import matplotlib.pyplot as plt
    plt.subplot( 211 )
    plt.plot( performer1ODF )
    plt.plot( performer2ODF )
    plt.subplot( 212 )
    plt.plot( correlation )
    plt.show()
    # Compute the entropy of the cross-correlation
    entropy = scipy.stats.entropy( correlation + 1e-100 )
    # Normalize entropy by the uniform distribution (upper bound)
    entropy = entropy/np.log( correlation.shape[0] )
    # Synchronization score is 1 - normalized entropy, because we want a high score to indicate high synchronization
    return np.clip( 1 - entropy, 0, np.inf )

