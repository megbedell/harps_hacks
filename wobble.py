import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.gridspec as gridspec
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

    pdb.set_trace()
    
    
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
    
    #PCA!
    # subtract off the mean values from every time series:
    a = order_time_par
    a_rvonly = a[:,:,1] # select RVs    
    a_rvonly -= np.repeat([np.mean(a_rvonly,axis=0)],48,axis=0)  #subtract off the mean value from each order time series
    a_rvonly /= np.repeat([np.sqrt(np.var(a_rvonly,axis=0))],48,axis=0)  #divide out the sqrt(variance) from each order time series
    a_all = np.reshape(a,(48,-1)) # select all Gaussian parameters
    a_all -= np.repeat([np.mean(a_all,axis=0)],48,axis=0)  #subtract off the mean value from each order time series
    a_all /= np.repeat([np.sqrt(np.var(a_all,axis=0))],48,axis=0)  #divide out the sqrt(variance) from each order time series
    a_norv = np.reshape(np.delete(a,1,axis=2),(48,-1)) # select all Gaussian parameters EXCEPT RV
    a_norv -= np.repeat([np.mean(a_norv,axis=0)],48,axis=0)  #subtract off the mean value from each order time series
    a_norv /= np.repeat([np.sqrt(np.var(a_norv,axis=0))],48,axis=0)  #divide out the sqrt(variance) from each order time series
    a_rvsig = np.reshape(np.delete(a,(0,3),axis=2),(48,-1)) # select RV mean & sigma only
    a_rvsig -= np.repeat([np.mean(a_rvsig,axis=0)],48,axis=0)  #subtract off the mean value from each order time series
    a_rvsig /= np.repeat([np.sqrt(np.var(a_rvsig,axis=0))],48,axis=0)  #divide out the sqrt(variance) from each order time series
    
    #do SVD on RVs only:
    u,s,v = np.linalg.svd(a_rvonly, full_matrices=False)
    #plot some eigenvalues:
    plt.scatter(np.arange(48),s**2)
    plt.title('Eigenvalues')
    plt.xlim(-1,50)
    plt.yscale('log')
    plt.ylim(0.5,5.0e2)
    plt.savefig('fig/eigenvalues_rv.png')
    plt.clf()
    #plot some eigenvectors:
    plt.plot(v[0,:],color='red',label='vector 1')
    plt.plot(v[1,:],color='blue',label='vector 2')
    plt.xlabel('Order #')
    plt.ylabel(r'$\Delta$ velocity (km/s)')
    plt.title('Eigenvectors')
    plt.legend()
    plt.savefig('fig/eigenvectors_rv.png')
    plt.clf()
    
    #do SVD on all Gaussian parameters:
    u,s,v = np.linalg.svd(a_all, full_matrices=False)
    #plot some eigenvalues:
    plt.scatter(np.arange(48),s**2)
    plt.title('Eigenvalues')
    plt.xlim(-1,50)
    plt.yscale('log')
    plt.ylim(10.0,1.0e4)
    plt.savefig('fig/eigenvalues_all.png')
    plt.clf()
    #plot some eigenvectors:
    plt.title('Eigenvectors')
    gs = gridspec.GridSpec(4, 1, hspace=0.2)
    for i,par in enumerate(['amplitude','mean RV','sigma RV','vertical offset']):
        ax = plt.subplot(gs[i])
        plt.plot(v[0][i::4],color='red',label='vector 1')
        plt.plot(v[1][i::4],color='blue',label='vector 2')
        #plt.ylim(np.min(v[0:2,i::4]), np.max(v[0:2,i::4]))
        plt.ylabel(r'$\Delta$'+par)
    plt.xlabel('Order #')
    #plt.tight_layout()
    plt.legend(loc='center right')
    plt.savefig('fig/eigenvectors_all.png')
    plt.clf()
    
    #do SVD on all Gaussian parameters EXCEPT the RVs:
    u,s,v = np.linalg.svd(a_norv, full_matrices=False)
    #plot some eigenvalues:
    plt.scatter(np.arange(48),s**2)
    plt.title('Eigenvalues')
    plt.xlim(-1,50)
    plt.yscale('log')
    plt.ylim(1.0,1.0e4)
    plt.savefig('fig/eigenvalues_norv.png')
    plt.clf()
    #plot some eigenvectors:
    plt.title('Eigenvectors')
    gs = gridspec.GridSpec(3, 1, hspace=0.15)
    for i,par in enumerate(['amplitude','sigma RV','vertical offset']):
        ax = plt.subplot(gs[i])
        ax.plot(v[0][i::3],color='red',label='vector 1')
        ax.plot(v[1][i::3],color='blue',label='vector 2')
        #plt.ylim(np.min(v[0:2,i::4]), np.max(v[0:2,i::4]))
        ax.set_ylabel(r'$\Delta$'+par)
    plt.xlabel('Order #')
    #plt.tight_layout()
    plt.legend(loc='center left')
    plt.savefig('fig/eigenvectors_norv.png')
    plt.clf()

    #do SVD on RV mean & sigma only:
    u,s,v = np.linalg.svd(a_rvsig, full_matrices=False)
    #plot some eigenvalues:
    plt.scatter(np.arange(48),s**2)
    plt.title('Eigenvalues')
    plt.xlim(-1,50)
    plt.yscale('log')
    plt.ylim(1.0e1,1.0e3)
    plt.savefig('fig/eigenvalues_rvsig.png')
    plt.clf()
    #plot some eigenvectors:
    plt.title('Eigenvectors')
    gs = gridspec.GridSpec(3, 2, hspace=0.15)
    ax = plt.subplot(gs[0,:])
    ax.plot(v[0][0::2],color='red',label='vector 1')
    ax.plot(v[1][0::2],color='blue',label='vector 2')
    ax.set_ylabel(r'$\Delta$ mean RV')
    ax = plt.subplot(gs[1,:])
    ax.plot(v[0][1::2],color='red',label='vector 1')
    ax.plot(v[1][1::2],color='blue',label='vector 2')
    ax.set_ylabel(r'$\Delta \sigma_{RV}$')
    ax = plt.subplot(gs[2,0])
    ax.scatter(v[0][0::2],v[0][1::2],color='red',label='vector 1')
    ax.scatter(v[1][0::2],v[1][1::2],color='blue',label='vector 2')
    ax.set_ylabel(r'$\Delta \sigma_{RV}$')
    ax.set_xlabel(r'$\Delta$ mean RV')
    ax2 = plt.subplot(gs[2,1])
    h,l=ax.get_legend_handles_labels()
    ax2.set_frame_on(False)
    ax2.axes.get_yaxis().set_visible(False)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.legend(h,l,loc='center',mode="expand") 
    

    #plt.xlabel('Order #')
    #plt.legend(loc='upper right')
    plt.savefig('fig/eigenvectors_rvsig.png')
    plt.clf()

    
    
