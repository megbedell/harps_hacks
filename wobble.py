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

def plot_timeseries(time,rv,rv_err,rv2=0):
    '''Plot RV timeseries (probably delete this later)
    '''
    t = astropy.time.Time(time, format='jd')
    rv = (rv - np.median(rv))*1.0e3
    plt.errorbar(t.datetime, rv, yerr=rv_err*1.0e3, fmt='o')
    if (len(rv2) == len(time)):
        rv2 = (rv2 - np.median(rv2))*1.0e3
        plt.errorbar(t.datetime, rv2, yerr=rv_err*1.0e3, fmt='o', color='red', ecolor='red')
    plt.ylabel('RV (m/s)')
    plt.xlim(t.datetime[0]-datetime.timedelta(days=100), t.datetime[-1]+datetime.timedelta(days=100))
    plt.show()   

if __name__ == "__main__":
    
    data_dir = "/Users/mbedell/Documents/Research/HARPSTwins/Results/"
    s = readsav(data_dir+'HIP54287_result.dat') 
 
    HIP54287 = rv_model.RV_Model()
    HIP54287.t = s.date - s.date[0]  # timeseries epochs
    HIP54287.get_data(s.files)  # fetch order-by-order RVs
    HIP54287.get_drift(s.files) # fetch instrument drifts
    HIP54287.get_wavepar(s.files) # fetch wavelength solution param
    wavepar = np.delete(HIP54287.wavepar,[71, 66, 57],axis=1)  # remove the orders that have no RVs
    HIP54287.set_param()    
    
    rv = np.median(HIP54287.data[:,:,1], axis=1) # median of every order RV
    rv -= HIP54287.drift*1e-3 # apply the drift correction
    print "pipeline's RV RMS = {0:.3f} m/s".format(np.std(s.rv)*1.0e3)
    print "unweighted median RV RMS = {0:.3f} m/s".format(np.std(rv)*1.0e3)

    #try weighted means
    rv_aweight = np.average(HIP54287.data[:,:,1], weights=abs(HIP54287.data[:,:,0]), axis=1) - HIP54287.drift*1e-3
    rv_a2weight = np.average(HIP54287.data[:,:,1], weights=(HIP54287.data[:,:,0])**2, axis=1) - HIP54287.drift*1e-3
    print "abs(a) weighted mean RV RMS = {0:.3f} m/s".format(np.std(rv_aweight)*1.0e3)
    print "a^2 weighted mean RV RMS = {0:.3f} m/s".format(np.std(rv_a2weight)*1.0e3)
        
    rv_aweightmed = []
    for i in range(len(HIP54287.t)):
        rv_aweightmed.append(weighted_median(HIP54287.data[i,:,1], ws=abs(HIP54287.data[i,:,0])))
    #rv_aweightmed = np.apply_along_axis(weighted_median,1,HIP54287.data[:,:,1])
    rv_aweightmed -= HIP54287.drift*1e-3 # apply the drift correction
    print "abs(a) weighted median RV RMS = {0:.3f} m/s".format(np.std(rv_aweightmed)*1.0e3)
    
    #fitting an individual line:
    wave_c = 6123.34
    #wave_c = 6043.2
    #wave_c = 6379.42
    rv_one = np.zeros(len(s.files))
    for i,f in enumerate(s.files):
        # read in the spectrum
        spec_file = str.replace(f, 'ccf_G2', 's1d')
        sp = fits.open(spec_file)
        header = sp[0].header
        n_wave = header['NAXIS1']
        crval1 = header['CRVAL1']
        cdelt1 = header['CDELT1']
        index = np.arange(n_wave)
        wave = crval1 + index*cdelt1
        spec = sp[0].data
        # fit the line
        rv_one[i] = read_harps.rv_oneline_fit(wave,spec,wave_c)
    print "RMS on single-line RV (wavlength {0:.1f} A): {1:.2f} m/s".format(wave_c,np.std(rv_one)*1.0e3)
    
        