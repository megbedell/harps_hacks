import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from astropy.io import fits
import pdb
from scipy.io.idl import readsav
import astropy.time
import datetime
import read_harps
import rv_model
from weighted_median import weighted_median
from scipy.stats.stats import pearsonr
import corner

class Mask:
    def __init__(self, file='data/G2.mas'):
        self.wi, self.wf, self.weight = np.loadtxt(file,unpack=True,dtype=np.float64)
        
    def pix_value(self, w, pix_size=0.01):
        '''
        Takes a pixel and evaluates its value multiplied by mask
        Parameters
        ----------
        w : float
        Starting wavelength of the pixel (Ang)
        pix_size : float, optional
        Wavelength span of the pixel, default 0.01 Ang
        
        Returns
        -------
        value of pixel * mask
        '''
        ind_ii = np.searchsorted(self.wi, w)  # next mask start after pixel start
        ind_if = np.searchsorted(self.wi, w+pix_size)  # next mask start after pixel end
        ind_fi = np.searchsorted(self.wf, w)  # next mask end after pixel start
        ind_ff = np.searchsorted(self.wf, w+pix_size)  # next mask end after pixel end
        if ind_ii == ind_if:
            if ind_fi == ind_ff:
                if ind_ii == ind_fi:  # pixel is entirely outside of mask
                    return 0.0
                else:  # pixel is entirely inside mask
                    return self.weight[ind_ff]
            else:  # beginning of pixel is inside mask
                return (self.wf[ind_fi] - w)/pix_size * self.weight[ind_fi]
        else:
            if ind_fi == ind_ff: # end of pixel is inside mask
                return (w+pix_size - self.wi[ind_ii])/pix_size * self.weight[ind_ii]
            else: # mask is entirely inside pixel
                return (self.wf[ind_fi] - self.wi[ind_ii])/pix_size * self.weight[ind_ii]
                

    def value(self, w):
        '''
        Takes a wavelength and evaluates the mask value there
        Parameters
        ----------
        w : float
        Wavelength (Ang)
        
        Returns
        -------
        mask value at the wavelength
        '''    
        if (w >= self.wi).any():
            ind = np.where(w >= self.wi)[0][-1]
            if w <= self.wf[ind]:
                return self.weight[ind]
            else:
                return 0.0
        else:
            return 0.0
            
    def construct(self, ws):
        '''
        Makes a mask for a given wavelength array
        '''
        ms = [self.value(w) for w in ws]
        return np.asarray(ms)
        
    def pix_values(self, ws, pix_size=0.01):
        '''
        Makes a mask for a given pixel array
        '''
        ms = [self.pix_value(w, pix_size=pix_size) for w in ws]
        return np.asarray(ms)

def doppler(wave, velocity):
    '''
    Takes a wavelength and a velocity (in km/s) and returns the shifted wavelength 
    in rest frame of an object moving at the given velocity.
    '''
    c = 299792.458 # speed of light in km/s
    wave = wave/(1.0 + velocity/c)
    return wave

def ccf(wave, flux, mask, velocity, pix_size=0.01):
    '''
    Get a cross-correlation function from a piece of spectrum.
    Parameters
    ----------
    wave : np.ndarray
    Wavelength values for spectral pixels
    flux : np.ndarray
    Normalized flux values for spectral pixels
    mask : Mask object
    cross-correlation mask to be used
    velocity : np.ndarray
    velocities at which to evaluate the CCF (in km/s)
    
    Returns
    -------
    ccf : np.ndarray
    CCF value
    '''
    ccf = np.zeros_like(velocity)
    #colors = iter(cm.gist_rainbow(np.linspace(0, 1, len(velocity))))
    for i,v in enumerate(velocity):
        wave_v = doppler(wave, v)  # shift mask wavelength range
        ccf[i] = np.sum(flux * mask.pix_values(wave_v, pix_size=pix_size))
        #color=next(colors)
        #plt.plot(wave, mask.construct(wave_v), color=color)
        #plt.plot(wave, flux*mask.pix_values(wave_v), color=color, ls='--')   
    #plt.scatter(wave,flux)
    #plt.xlim([6245.9,6246.9])
    #plt.show()
    
    return ccf
    
    