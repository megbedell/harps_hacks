import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import pdb
import read_harps
import rv_model
from weighted_median import weighted_median
from scipy.interpolate import UnivariateSpline as fit_spline

def continuum_1d(wave, spec, threshold=20, n_smooth=100):
    '''Fits and divides out a polynomial continuum normalization.

    Parameters
    ----------
    wave : np.ndarray
    wavelength (in Angstroms)
    spec : np.ndarray
    spec value
    threshold : int (0-100)
    percentile level at which to cut off non-continuum points
    (higher is stricter & will lead to fewer points in the fit)
    n_smooth : number of neighbor pixels to "smooth over" when picking continuum points

    Returns
    -------
    norm_spec : np.ndarray
    normalized spec value
    '''
    
    # mask out the absorption lines:
    mask = np.zeros_like(pix)
    
    for i in range(len(spec)):
        if not mask[i]: # if i is unmasked...
            neighbor_lo = max(0, i - n_smooth/2)
            neighbor_hi = min(neighbor_lo + n_smooth, len(spec))
            print "pixel {0} value: {1:.1f} // 20th percent of neighbors: {2:.1f}".format(i,spec[i],
                    np.percentile(ma.array(spec,mask=mask)[neighbor_lo:neighbor_hi], threshold))
            if spec[i] < np.percentile(ma.array(spec,mask=mask)[neighbor_lo:neighbor_hi], threshold):
                print "pixel masked."
                mask[i-2:i+3] = 1.0  # mask out five pixels centered on i
    
    
    
    
            
    
    # fit a spline:
    spline = fit_spline(ma.array(wave,mask=mask), ma.array(spec,mask=mask))
    continuum = spline(wave)
    
    # normalize & return:
    norm_spec = spec/continuum
    
    plt.scatter(wave,spec,color='red')
    plt.scatter(ma.array(wave,mask=mask),ma.array(spec,mask=mask),color='blue')
    plt.plot(wave,continuum)
    plt.savefig('continuum.png')
    return norm_spec
    
