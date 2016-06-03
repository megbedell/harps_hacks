import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pdb
from scipy.io.idl import readsav
import read_harps

class RV_Model:
    """A white-noise model considering RVs from each echelle order 
    as independent measurements."""
    
    def __init__(self,data=None,param=None):
        self.data = data
        self.param = param
    
    def __call__(self):
        return get_likelihood(self.data,self.param)
    
    def get_data(self,datafiles):
        """input: a list of all HARPS pipeline CCF data product filenames.
        fits Gaussians to the CCFs and outputs RV per order
        output self.data: shape n_epochs x 69 orders x 4 Gaussian fit param"""
        data = np.zeros((len(datafiles), 69, 4))
        for i,f in enumerate(datafiles):
            velocity, ccf, pipeline_rv = read_harps.read_ccfs(f)
            order_par = read_harps.rv_gaussian_fit(velocity, ccf, n_points=20) # chose n_points=20 from the median order RMS plot sent to Hogg on May 13 
            data[i,:,:] = order_par
        
        self.data = data

    def set_param(self,v0=0.0,a=0.0,planetpar=None):
        """set model parameters for the stellar RV.
        can only handle a linear trend... so far!"""
        self.param = {"v0":v0, "a":a}

def get_likelihood(data,param):
    print "Who knows?"