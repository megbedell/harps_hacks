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
    
    '''''
    corner_par = np.reshape(Star.data,(-1,4))
    corner_par[:,0] /= -1.0e6
    corner_par[:,3] /= 1.0e6
    corner_labels = ['CCF depth (millions)','CCF mean',r'CCF $\sigma$','CCF y-offset (millions)']
    corner.corner(corner_par,labels=corner_labels,plot_contours=False,title_kwargs={"fontsize": 12})
    plt.savefig('fig/ccf_corner.png')
    
    


    #try weighted means
    rv_aweight = np.average(Star.data[:,:,1], weights=abs(Star.data[:,:,0]), axis=1) - Star.drift*1e-3
    rv_a2weight = np.average(Star.data[:,:,1], weights=(Star.data[:,:,0])**2, axis=1) - Star.drift*1e-3
    print "abs(a) weighted mean RV RMS = {0:.3f} m/s".format(np.std(rv_aweight)*1.0e3)
    print "a^2 weighted mean RV RMS = {0:.3f} m/s".format(np.std(rv_a2weight)*1.0e3)
        
    rv_aweightmed = []
    for i in range(len(Star.t)):
        rv_aweightmed.append(weighted_median(Star.data[i,:,1], ws=abs(Star.data[i,:,0])))
    #rv_aweightmed = np.apply_along_axis(weighted_median,1,Star.data[:,:,1])
    rv_aweightmed -= Star.drift*1e-3 # apply the drift correction
    print "abs(a) weighted median RV RMS = {0:.3f} m/s".format(np.std(rv_aweightmed)*1.0e3)
    
    #try refitting RVs with central CCF points excluded
    rv_all = np.copy(Star.data[:,:,1])
    Star.get_data(s.files, n_points=20, mask_inner=4)
    rv_w = np.copy(Star.data[:,:,1])  # RVs from CCF wings
    Star.get_data(s.files, n_points=4, mask_inner=0)
    rv_c = np.copy(Star.data[:,:,1])  # RVs from CCF centers
    rv_diff = (rv_w - rv_c)/2.0
    rv_avg = (rv_w + rv_c)/2.0
    colors = iter(cm.gist_rainbow(np.linspace(0, 1, 69)))    
    for i in range(69): plt.scatter(rv_avg[:,i], rv_diff[:,i], c=next(colors))
    plt.ylabel(r'$(RV_w - RV_c)$/2')
    plt.xlabel(r'$(RV_w + RV_c)$/2')
    sm = plt.cm.ScalarMappable(cmap=cm.rainbow, norm=plt.Normalize(vmin=0, vmax=69))
    sm._A = []
    plt.colorbar(sm)
    #plt.xlim([-0.5,9.5])
    plt.savefig('fig/CCF_wingsvscenter.png')
    plt.clf()
    
    # multiple linear regression with central/outer CCF fits
    m = 47  # a high-amplitude CCF order
    v_m = rv_all[:,m] # time-series RVs for this order
    N = len(Star.t)
    n_pred = 2 # number of predictors, including intercept term
    A_m = np.ones((N,n_pred)) # design matrix for this order
    #A_m[:,1:5] = Star.wavepar[:,m,:]
    A_m[:,1] = rv_diff[:,m]
    x_m = np.zeros((N,n_pred))
    l = 0.0  # regularization parameter
    reg_matrix = np.identity(n_pred)  # regularization...
    reg_matrix[0] = 0                 # ... matrix
    v_pred = np.zeros(N) # predicted RVs
    for i in range(N):
        A_noi = np.delete(A_m, i, axis=0)
        v_noi = np.delete(v_m, i)
        x_noi = np.linalg.solve(np.dot(A_noi.T,A_noi)+l*reg_matrix, \
                np.dot(A_noi.T,v_noi)) # best-fit coeffs excluding i-th epoch
        x_m[i,:] = x_noi
        v_pred[i] = np.dot(A_m[i,:], x_noi) # leave-one-out regression prediction for this epoch
        
    
    plt.plot((v_m - v_m[0])*1e3, color='red', label='observed velocities')
    plt.plot((v_pred - v_m[0])*1e3, color='blue', label='predicted from center/wing CCF differences')
    plt.xlabel('Epoch #')
    plt.ylabel('RV (m/s)')
    plt.legend()
    #plt.title('linear regression with regularization param = {0}'.format(l))
    plt.savefig('fig/regression_o{0}ccf.png'.format(m))
    plt.clf()
    
    for i in range(n_pred):
        plt.plot(x_m[:,i]/np.max(abs(x_m[:,i])), label=r'x_{0}'.format(i))
    plt.legend()
    plt.xlabel('Epoch #')
    plt.ylabel('Normalized coefficient')
    plt.title('Leave-one-out Regression Coefficient Values')
    plt.ylim([-1.1,1.1])
    plt.savefig('fig/regression_coeff_o{0}ccf.png'.format(m))
    plt.clf()
    
    print "RMS of order {0} RVs before regression: {1:.2f} m/s".format(m, np.std(v_m)*1.0e3)
    print "RMS of order {0} residuals to regression: {1:.2f} m/s".format(m, np.std(v_m - v_pred)*1.0e3)
    
    plt.scatter(rv_diff[:,m], rv_all[:,m])
    plt.ylabel('RV from full CCF (km/s)')
    plt.xlabel(r'$(RV_w - RV_c)$/2')
    colors = iter(cm.gist_rainbow(np.linspace(0, 1, N)))    
    for i in range(N):
        x = np.arange(-0.05,0.05,0.001)
        fit = x_m[i,0] + x_m[i,1]*x
        plt.plot(x, fit, c=next(colors))
    sm = plt.cm.ScalarMappable(cmap=cm.rainbow, norm=plt.Normalize(vmin=0, vmax=N))
    sm._A = []
    plt.colorbar(sm)
    plt.xlim([-0.04,0.02])
    plt.savefig('fig/regression_o{0}fit.png'.format(m))
    plt.clf()
    
    
    '''''
    plt.clf()
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

        