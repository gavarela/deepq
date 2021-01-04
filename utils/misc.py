## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Just a miscellaneous function(s)
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np


# Interpolate
def interp(xs, Y, maxX):
    xs = np.array(xs)
    r  = (maxX-1)/(len(Y)-1)
    i  = (xs/r).astype(int)
    Y  = np.array(list(Y) + [Y[-1]])
    return Y[i] + (xs/r - i)*(Y[i+1]-Y[i])

