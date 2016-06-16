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
    a = np.copy(order_time_par)
    a_rvonly = a[:,:,1] # select RVs    
    a_rvonly -= np.repeat([np.mean(a_rvonly,axis=0)],48,axis=0)  #subtract off the mean value from each order time series
    
    # does pipeline RV correlate with the sidereal day?
    sday = 0.99726958
    date_fold = HIP54287.t % sday
    plt.scatter(date_fold, (s.rv-np.mean(s.rv))*1e3)
    plt.ylabel("Pipeline RV (m/s)")
    plt.xlabel("Date mod 1 sidereal day")
    plt.xlim(-0.05,0.25)
    plt.savefig("fig/sidereal_day.png")
    print 'Pearson R for RV & date mod sidereal = {d[0]}, p-val = {d[1]}'.format(d=pearsonr((s.rv-np.mean(s.rv))*1e3, date_fold))
    plt.clf()
    
    # how do the wavelength solution coefficients compare with RV offset per order?
    wavepar = np.delete(HIP54287.wavepar,[71, 66, 57],axis=1)  # remove the orders that have no RVs
    wavepar_avg = np.mean(wavepar,axis=0)
    data_avg = np.mean(HIP54287.data,axis=0)
    plt.scatter(data_avg[:,1],wavepar_avg[:,0])
    plt.xlabel('order offset (km/s)')
    plt.ylabel('Wavelength parameter 0')
    plt.ylim(np.min(wavepar_avg[:,0])*0.5,np.max(wavepar_avg[:,0])*1.5)
    plt.savefig('fig/wavepar0.png')
    plt.clf()
    plt.scatter(data_avg[:,1],wavepar_avg[:,1])
    plt.xlabel('order offset (km/s)')
    plt.ylabel('Wavelength parameter 1')
    plt.ylim(np.min(wavepar_avg[:,1])*0.5,np.max(wavepar_avg[:,1])*1.5)
    plt.savefig('fig/wavepar1.png')
    plt.clf()
    plt.scatter(data_avg[:,1],wavepar_avg[:,2])
    plt.xlabel('order offset (km/s)')
    plt.ylabel('Wavelength parameter 2')
    plt.ylim(np.min(wavepar_avg[:,2])*1.5,np.max(wavepar_avg[:,2])*0.5)
    plt.savefig('fig/wavepar2.png')
    plt.clf()
    plt.scatter(data_avg[:,1],wavepar_avg[:,3])
    plt.xlabel('order offset (km/s)')
    plt.ylabel('Wavelength parameter 3')
    plt.ylim(np.min(wavepar_avg[:,3])*1.5,np.max(wavepar_avg[:,3])*0.5)
    plt.savefig('fig/wavepar3.png')
    plt.clf()
    
    # how do the coefficients change over time?
    order = 39  # a high-amplitude CCF order
    plt.scatter(HIP54287.t, wavepar[:,order,0])
    plt.xlabel('Relative J.D.')
    plt.ylabel('Wavelength parameter 0')
    scale = (wavepar[:,order,0].max()-wavepar[:,order,0].min())/2.0
    plt.ylim(wavepar[:,order,0].min()-scale,wavepar[:,order,0].max()+scale)
    plt.savefig('fig/wavepar0_time.png')
    plt.clf()
    plt.scatter(HIP54287.t, wavepar[:,order,1])
    plt.xlabel('Relative J.D.')
    plt.ylabel('Wavelength parameter 1')
    scale = (wavepar[:,order,1].max()-wavepar[:,order,1].min())/2.0
    plt.ylim(wavepar[:,order,1].min()-scale,wavepar[:,order,1].max()+scale)
    plt.savefig('fig/wavepar1_time.png')
    plt.clf()    
    plt.scatter(HIP54287.t, wavepar[:,order,2])
    plt.xlabel('Relative J.D.')
    plt.ylabel('Wavelength parameter 2')
    scale = (wavepar[:,order,2].max()-wavepar[:,order,2].min())/2.0
    plt.ylim(wavepar[:,order,2].min()-scale,wavepar[:,order,2].max()+scale)
    plt.savefig('fig/wavepar2_time.png')
    plt.clf()    
    plt.scatter(HIP54287.t, wavepar[:,order,3])
    plt.xlabel('Relative J.D.')
    plt.ylabel('Wavelength parameter 3')
    scale = (wavepar[:,order,3].max()-wavepar[:,order,3].min())/2.0
    plt.ylim(wavepar[:,order,3].min()-scale,wavepar[:,order,3].max()+scale)
    plt.savefig('fig/wavepar3_time.png')
    plt.clf()
    
    
