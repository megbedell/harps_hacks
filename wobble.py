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
    HIP54287.get_wavepar(s.files) # fetch wavelength solution param
    HIP54287.set_param()    
    
    order_time_par = HIP54287.data        
    rv = np.median(order_time_par[:,:,1], axis=1) # median of every order RV
    print "pipeline's RV RMS = {0:.3f} m/s".format(np.std(s.rv)*1.0e3)
    print "unweighted median RV RMS = {0:.3f} m/s".format(np.std(rv)*1.0e3)
    
    
    #try weighted means
    rv_aweight = np.average(order_time_par[:,:,1], weights=abs(order_time_par[:,:,0]), axis=1)
    rv_a2weight = np.average(order_time_par[:,:,1], weights=(order_time_par[:,:,0])**2, axis=1)
    print "abs(a) weighted mean RV RMS = {0:.3f} m/s".format(np.std(rv_aweight)*1.0e3)
    print "a^2 weighted mean RV RMS = {0:.3f} m/s".format(np.std(rv_a2weight)*1.0e3)
    
    rv_aweightmed = []
    for i in range(len(HIP54287.t)):
        rv_aweightmed.append(weighted_median(order_time_par[i,:,1], ws=abs(order_time_par[i,:,0])))
    #rv_aweightmed = np.apply_along_axis(weighted_median,1,order_time_par[:,:,1])
    print "abs(a) weighted median RV RMS = {0:.3f} m/s".format(np.std(rv_aweightmed)*1.0e3)
    

    # subtract off the mean values from every time series:
    a = order_time_par
    a_rvonly = a[:,:,1] # select RVs    
    a_rvonly -= np.repeat([np.mean(a_rvonly,axis=0)],48,axis=0)  #subtract off the mean value from each order time series
    
    #  let's try some dumb stuff
    # does pipeline RV correlate with airmass?
    print 'Pearson R for RV & airmass = {d[0]}, p-val = {d[1]}'.format(d=pearsonr((s.rv-np.mean(s.rv))*1e3, s.airm))
    plt.scatter((s.rv-np.mean(s.rv))*1e3, s.airm)
    plt.ylabel("Airmass")
    plt.xlabel("Pipeline RV (m/s)")
    plt.savefig("fig/airmass.png")
    plt.clf()
    # does pipeline RV correlate with SNR?
    print 'Pearson R for RV & SNR = {d[0]}, p-val = {d[1]}'.format(d=pearsonr((s.rv-np.mean(s.rv))*1e3, s.snr))
    plt.scatter((s.rv-np.mean(s.rv))*1e3, s.snr)
    plt.ylabel("SNR")
    plt.xlabel("Pipeline RV (m/s)")
    plt.savefig("fig/snr.png")
    plt.clf()
    # does pipeline RV correlate with seeing?
    seeing = np.arange(len(s.files), dtype=np.float)
    for i in np.nditer(seeing, op_flags=['readwrite']):
        sp = fits.open(s.files[int(i)])
        header = sp[0].header
        i[...] = np.mean([header['HIERARCH ESO TEL AMBI FWHM START'], header['HIERARCH ESO TEL AMBI FWHM END']])
    rv_seeing = np.delete(s.rv,np.where(seeing == -1))  #remove epochs with no seeing measured
    seeing = np.delete(seeing,np.where(seeing == -1))
    
    print 'Pearson R for RV & seeing = {d[0]}, p-val = {d[1]}'.format(d=pearsonr((rv_seeing-np.mean(rv_seeing))*1e3, seeing))
    plt.scatter((rv_seeing-np.mean(rv_seeing))*1e3, seeing)
    plt.ylabel("Seeing")
    plt.xlabel("Pipeline RV (m/s)")
    plt.savefig("fig/seeing.png")
    plt.clf()
    
    # does pipeline RV correlate with the sidereal day?
    sday = 0.99726958
    date_fold = HIP54287.t % sday
    plt.scatter(date_fold, (s.rv-np.mean(s.rv))*1e3)
    plt.ylabel("Pipeline RV (m/s)")
    plt.xlabel("Date mod 1 sidereal day")
    plt.savefig("fig/sidereal_day.png")
    plt.clf()
    print 'Pearson R for RV & date mod sidereal = {d[0]}, p-val = {d[1]}'.format(d=pearsonr((s.rv-np.mean(s.rv))*1e3, date_fold))

    
    
    
