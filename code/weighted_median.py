import numpy as np
from scipy.interpolate import interp1d

def weighted_median(xs, ws=None):
    if ws is None:
        ws = np.ones_like(xs)
    totalw = np.sum(ws)
    sindx = np.argsort(xs) # expensive                                                                                                 
    cs = np.cumsum(ws[sindx]) / totalw
    symcumsum = 0.5 * (cs + np.append(0., cs[0:-1]))
    #print(symcumsum, xs[sindx])
    return (interp1d(symcumsum, xs[sindx]))(0.5)
