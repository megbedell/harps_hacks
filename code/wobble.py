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
import xcorrelate
from weighted_median import weighted_median
from scipy.stats.stats import pearsonr
import corner

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

def rv_oneline(wave_c,filenames):
    '''Measure RV timeseries from a single line.
    Parameters
    ----------
    wave_c : float
    starting guess of the line's central wavelength
    filenames : string (list)
    list of CCF files
    
    Returns
    -------
    rv : np.float64 (list)
    RV shift of the line at each epoch
    '''
    rv_one = np.zeros(len(filenames))
    for i,f in enumerate(filenames):
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
    return rv_one
     
if __name__ == "__main__":
    
    data_dir = "/Users/mbedell/Documents/Research/HARPSTwins/Results/"
    s = readsav(data_dir+'HIP22263_result.dat') 
    
    #s = read_harps.read_csv('data/HIP22263/HIP22263.csv')
 
    Star = rv_model.RV_Model()
    Star.t = s.date - s.date[0]  # timeseries epochs
    Star.get_data(s.files)  # fetch order-by-order RVs
    Star.get_drift(s.files) # fetch instrument drifts
    Star.get_wavepar(s.files) # fetch wavelength solution param
    wavepar = np.delete(Star.wavepar,[71, 66, 57],axis=1)  # remove the orders that have no RVs
    Star.set_param()    
    
    rv = np.median(Star.data[:,:,1], axis=1) # median of every order RV
    rv -= Star.drift*1e-3 # apply the drift correction
    print "pipeline's RV RMS = {0:.3f} m/s".format(np.std(s.rv)*1.0e3)
    print "unweighted median RV RMS = {0:.3f} m/s".format(np.std(rv)*1.0e3)
    
    G2mask = xcorrelate.Mask(file='data/G2.mas')
    
    # testing continuum normalization:

    
    #plt.xlim([6245.9,6246.9])
    for i,f in enumerate(s.files[0:2]):
        # read in the spectrum
        spec_file = str.replace(f, 'ccf_G2', 's1d')
        wave, spec = read_harps.read_spec(spec_file)
        #spec /= np.percentile(spec,99)  # really dumb continuum normalization
        plt.plot(wave,spec)
        plt.xlim([4900.0,5000.0])
    plt.show()
    plt.clf()
    
    fig = plt.figure()
    wave1, spec1 = read_harps.read_spec(str.replace(s.files[110], 'ccf_G2', 's1d'))
    wave2, spec2 = read_harps.read_spec(str.replace(s.files[17], 'ccf_G2', 's1d'))
    wave = np.arange(np.max([wave1[0],wave2[0]]),np.min([wave1[-1],wave2[-1]]),0.01)
    spec1 = np.interp(wave, wave1, spec1)
    spec2 = np.interp(wave, wave2, spec2)

    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(wave,spec1)
    ax1.plot(wave,spec2)
    ax1.set_xlim([4900.0,5000.0])
    ax1.set_ylim([0,25000])
    #ax.set_xticklabels( () )
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)
    ax2.plot(wave, spec1/spec2)
    ax2.set_xlim([4900.0,5000.0])
    ax2.set_ylim([0.68,0.8])
    fig.subplots_adjust(hspace=0.05)
    plt.show()
    
    
    #mask = [G2mask.value(w) for w in wave]
    #plt.plot(wave, mask)
    #plt.show()
    

    #test_ind = np.where((wave > 6245.9) & (wave < 6246.9))
    #test_ind = np.where((wave > 4950.0) & (wave < 5000.0))  # order ind = 38
    #velocity = np.arange(10.0,35.0,0.1)
    #ccf_test = xcorrelate.ccf(wave[test_ind], spec[test_ind], G2mask, velocity)
    
    #plt.plot(velocity,ccf_test)
    #plt.xlabel('RV (km/s)')
    #plt.ylabel('CCF value')
    #plt.savefig('fig/ccf.png')
    
    
    
    

    #fitting an individual line:
    #wave_c = 6123.34
    #wave_c = 6043.2
    #wave_c = 6379.42
    #rv_one = rv_oneline(wave_c, s.files)

    #print "RMS on single-line RV (wavelength {0:.1f} A): {1:.2f} m/s".format(wave_c,np.std(rv_one)*1.0e3)

        