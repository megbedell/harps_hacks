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
        self.wi, self.wf, self.weight = np.loadtxt(file,unpack=True)
        
    def value(self, w):
        '''
        Takes a wavelength and evaluates its mask value 
        '''
        if (w >= self.wi).any():
            ind = np.where(w >= self.wi)[0][-1]
            if w <= self.wf[ind]:
                return self.weight[ind]
            else:
                return 0.0
        else:
            return 0.0
