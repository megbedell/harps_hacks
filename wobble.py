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
    HIP54287.set_param()    
    
    order_time_par = HIP54287.data        
    rv = np.nanmedian(order_time_par[:,:,1], axis=1) # median of every order RV
    print "pipeline's RV RMS = {0:.3f} m/s".format(np.std(s.rv)*1.0e3)
    
    
    #try weighted means
    rv_aweight = np.average(order_time_par[:,:,1], weights=abs(order_time_par[:,:,0]), axis=1)
    rv_a2weight = np.average(order_time_par[:,:,1], weights=(order_time_par[:,:,0])**2, axis=1)
    print "abs(a) weighted RV RMS = {0:.3f} m/s".format(np.std(rv_aweight)*1.0e3)
    print "a^2 weighted RV RMS = {0:.3f} m/s".format(np.std(rv_a2weight)*1.0e3)
    
    par_meansub =  order_time_par[:,:,1] - np.repeat([np.average(order_time_par[:,:,1], axis=0)],48,axis=0)
    rv_meansub_aweight = np.average(par_meansub, weights=abs(order_time_par[:,:,0]), axis=1)
    print "abs(a) weighted, mean-subtracted RV RMS = {0:.3f} m/s".format(np.std(rv_meansub_aweight)*1.0e3)   
    
    # subtract off the mean values from every time series:
    a = order_time_par
    a_rvonly = a[:,:,1] # select RVs    
    a_rvonly -= np.repeat([np.mean(a_rvonly,axis=0)],48,axis=0)  #subtract off the mean value from each order time series
   
    # plot time series of each order
    fig = plt.figure()
    gs = gridspec.GridSpec(2,1,height_ratios=[5,1],hspace=0.05)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    colors = iter(cm.gist_rainbow(np.linspace(0, 1, 69)))
    for i in range(69):
        ax1.plot(a_rvonly[:,i]+i/100.0, color=next(colors))
    ax2.plot((s.rv-np.mean(s.rv)), color='black')
    ax1.set_ylabel('RV + offset (km/s)')
    ax2.set_ylabel('RV (km/s)')
    ax2.set_xlabel('Epoch #')
    #sm = plt.cm.ScalarMappable(cmap=cm.rainbow, norm=plt.Normalize(vmin=0, vmax=69))
    #sm._A = []
    #plt.colorbar(sm)
    plt.savefig('fig/timeseries_orders')
    plt.clf()

    
    
