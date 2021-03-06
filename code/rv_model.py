import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pdb
from scipy.io.idl import readsav
import read_harps

class RV_Model:
    """A white-noise model considering RVs from each echelle order 
    as independent measurements."""
    
    def __init__(self,t=None,data=None,param=None,model=None):
        self.t = t
        self.data = data
        self.param = param
        self.model = model
    
    def __call__(self):
        return self.get_lnprob()
    
    def get_data(self,datafiles, n_points=20, mask_inner=0, debug=False):
        """input: a list of all HARPS pipeline CCF data product filenames.
        fits Gaussians to the CCFs and outputs RV per order
        for keyword meanings, see read_harps.rv_gaussian_fit
        output self.data: shape n_epochs x 69 orders x 4 Gaussian fit param"""
        data = np.zeros((len(datafiles), 69, 4))
        for i,f in enumerate(datafiles):
            velocity, ccf, pipeline_rv = read_harps.read_ccfs(f)
            order_par = read_harps.rv_gaussian_fit(velocity, ccf, n_points=n_points, mask_inner=mask_inner, debug=debug) # chose n_points=20 from the median order RMS plot sent to Hogg on May 13 
            data[i,:,:] = order_par
        
        self.data = data
    
    def get_drift(self,datafiles):
        """input: a list of all HARPS pipeline CCF data product filenames.
        saves the instrumental drift as determined by simultaneous reference
        output self.drift: shape n_epochs array of drift"""
        drift = np.zeros(len(datafiles))
        for i,f in enumerate(datafiles):
            drift[i] = read_harps.read_drift(f)
        self.drift = drift
        
    def get_wavepar(self,datafiles):
        """input: a list of all HARPS pipeline CCF data product filenames.
        reads headers and outputs wavelength solution coefficients
        output self.wavepar: shape n_epochs x 72 orders x 4 wavelength param"""
        wavepar = np.zeros((len(datafiles), 72, 4))
        for i,f in enumerate(datafiles):
            wavepar[i,:,:] = read_harps.read_wavepar(f)
        
        self.wavepar = wavepar
        

    def set_param(self,b=None,c=None,order_offset=None,v0=0.0,linear=0.0,planetpar=None):
        """set model parameters for the RV & uncertainties.
        can only handle a linear trend... so far!"""
        if b is None:
            n_epochs = np.shape(self.data)[0]
            b = np.ones(n_epochs) # error per epoch
        if c is None:
            c = np.ones(69) # error per order
        if order_offset is None:
            order_offset = np.zeros(69) # RV offset per order
        self.param = {"b":b, "c":c, "order_offset":order_offset, "v0":v0, "linear":linear}
        
    def get_lnprob(self, param=None):
        if param is None:
            param = self.param
        n_epochs = np.shape(self.data)[0]
        sig = np.repeat([param['b']],69,axis=0).T * np.repeat([param['c']],n_epochs,axis=0) # [n_epochs, 69] array of errors
        
        rv_star = np.zeros(n_epochs) + param['v0'] + param['linear']*self.t # [n_epochs] timeseries of modelled stellar RV (f(t_n; theta) in notes)
        
        obs = self.data[:,:,1] # central RVs only
        lnprob_all = -0.5 * (obs - np.repeat([param['order_offset']],n_epochs,axis=0) - \
                        np.repeat([rv_star],69,axis=0).T)**2/sig**2 - 0.5*np.log(sig**2)
        lnprob = np.sum(lnprob_all)
        
        return lnprob