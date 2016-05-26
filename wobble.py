import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from astropy.io import fits
from scipy.optimize import curve_fit
import pdb
from scipy.io.idl import readsav
import astropy.time
import datetime


def read_ccfs(filename):
    '''Read a HARPS CCF file from the ESO pipeline

    Parameters
    ----------
    filename : string
    name of the fits file with the data

    Returns
    -------
    velocity : np.ndarray
    velocity (in km/s)
    ccf : np.ndarray
    ccf value
    rv : float
    the pipeline-delivered RV **without drift correction applied** (in km/s)
    '''
    sp = fits.open(filename)
    header = sp[0].header

    # get the relevant header info
    n_vels = header['NAXIS1']
    n_orders = header['NAXIS2']
    crval1 = header['CRVAL1']
    cdelt1 = header['CDELT1']
    
    # construct an (n_orders, n_vels) array of velocities
    index = np.arange(n_vels)
    velocity_one = crval1 + index*cdelt1
    velocity = np.tile(velocity_one, (n_orders,1))

    # get the ccf values
    ccf = sp[0].data
    
    # get the header RV
    rv = header['HIERARCH ESO DRS CCF RV']
        
    return velocity, ccf, rv

def gauss_function(x, a, x0, sigma, offset):
    '''it's a Gaussian.'''
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + offset

def parabola_min(x, y):
    '''Take three data points (x,y), fit a parabola, and return the minimum
    
    Parameters
    ----------
    x : np.ndarray, len = 3
    y : np.ndarray, len = 3
    
    Returns
    -------
    x_min : float
    location of the parabola minimum
    '''
    M = np.vstack([x**2, x, np.ones(len(x))]).T
    a,b,c = np.linalg.solve(M, y)
    x_min = -b/(2.0*a)
    return x_min
    

def plot_ccfs(velocity, ccf, pipeline_rv, custom_rvs, file_out='all_ccfs.png', calc_sum=False):
    '''Make a multipanel plot of all CCFs, order-by-order

    Parameters
    ----------
    velocity : np.ndarray
    velocity (in km/s)
    ccf : np.ndarray
    ccf value
    pipeline_rv : float
    the HARPS-determined RV from the FITS header (km/s)
    custom_rvs : np.ndarray
    custom-determined RVs for each order (km/s)
    file_out : string (optional keyword)
    name for the output plot; default- 'all_ccfs.png'
    calc_sum : boolean (optional keyword)
    if True, the co-added CCF from all orders is calculated

    Returns
    -------
    none
    '''
    # set up the figure
    fig = plt.figure(figsize=(8,20))
    fig.subplots_adjust(bottom=0.03, left=0.025, top = 0.975, right=0.975, hspace=0.5)
    majorLocator   = MultipleLocator(10)
    minorLocator   = MultipleLocator(5)
    # step through CCF orders
    ccf_rv = np.array([])
    j=1
    for i in np.arange(73):
        # skip the bad orders
        if i in [71, 66, 57]:
            continue
        # plot the CCF for this order
        ax = fig.add_subplot(14,5,j)
        j += 1
        ax.plot(velocity[i],ccf[i])
        height = max(ccf[i]) - min(ccf[i])
        plt.ylim(min(ccf[i]) - height*0.1, max(ccf[i]) + height*0.1)
        ax.yaxis.set_ticks([])
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.tick_params(labelsize=6, length=3)
        # find the central value and mark it        
        popt, pcov = curve_fit(gauss_function, velocity[i], ccf[i], p0 = [-height, np.median(velocity[i]), 3.0, max(ccf[i])])
        ccf_rv = np.append(ccf_rv,popt[1])
        ax.axvline(pipeline_rv, color='red', linestyle='-', linewidth=0.3)
        ax.axvline(ccf_rv[-1], color='blue', linestyle='--', linewidth=0.3)
        ax.axvline(custom_rvs[i], color='green', linestyle='--', linewidth=0.3)
        # add labels
        if i == 72:
            ax.set_title('HARPS Combined CCF\nRV {0:.5f} km/s'.format(ccf_rv[-1]), fontsize=7)
            continue
        ax.set_title('Order {0}\nRV {1:.3f} km/s'.format(i+1, ccf_rv[-1]), fontsize=7)
    
    
    if calc_sum:
        # overplot the actual co-added CCF on the final panel & write out the RV that gives
        # (in principle this should be the same as the final 'order' from the pipeline,
        #  but HARPS pipeline apparently excludes the first order when making co-added CCF.)
        ccf_sum = np.sum(ccf[:-1,:], axis=0)
        ax.plot(velocity[-1],ccf_sum,color='green')
        height = max(ccf_sum) - min(ccf_sum)
        popt, pcov = curve_fit(gauss_function, velocity[-1], ccf_sum, p0 = [-height, np.median(velocity[-1]), 3.0, max(ccf_sum)])
        ccf_sum_rv = popt[1]
        ax.axvline(ccf_sum_rv, color='green', linestyle='--', linewidth=0.3)
        ax.set_title('Co-added CCF\nRV {0:.5f} km/s'.format(ccf_sum_rv), fontsize=7)

        
    plt.figtext(0.5,0.01,'Pipeline RV = {0:.5f} km/s'.format(pipeline_rv), horizontalalignment='center', color='red')
    
    fig.savefig(file_out, dpi=400)



def rv_parabola_fit(velocity, ccf):
    '''Read in the pipeline CCF data product and return polynomial-fitted RV for each order
    
    Parameters
    ----------
    velocity : np.ndarray
    velocity (in km/s)
    ccf : np.ndarray
    ccf value

    Returns
    -------
    order_rvs : np.ndarray
    the RV minimum of each order's ccf (km/s)
    '''
    order_rvs = np.zeros(ccf.shape[0])
    for i in xrange(ccf.shape[0]):
        if i in [71, 66, 57]:  # disregard the bad orders
            order_rvs[i] = np.nan
            continue
        ind_min = np.argmin(ccf[i])
        x = velocity[i][ind_min-1:ind_min+2]  # select just the nearest 3 pts to fit
        y = ccf[i][ind_min-1:ind_min+2]
        order_rvs[i] = parabola_min(x,y)  # find the RV minimum using parabolic interpolation
    return order_rvs
    
def rv_gaussian_fit(velocity, ccf, n_points=20, debug=False):
    '''Read in the pipeline CCF data product and return Gaussian-fitted RV for each order
    
    Parameters
    ----------
    velocity : np.ndarray
    velocity (in km/s)
    ccf : np.ndarray
    ccf value
    n_points : int
    total number of points on either side of the minimum to fit, must be >= 2

    Returns
    -------
    order_par : np.ndarray
    best-fit Gaussian parameters for each order: (amplitude, mean, sigma, offset)
    '''
    if (n_points < 1):
        print "Cannot fit a Gaussian to < 4 points! Try n_points = 2"
        return None
    order_par = np.zeros((ccf.shape[0],4))
    for i,order in enumerate(ccf):
        if i in [71, 66, 57]:  # disregard the bad orders
            order_par[i,:] = np.nan
            continue
        ind_min = np.argmin(order)
        ind_range = np.arange(n_points*2+1) + ind_min - n_points
        if (ind_range > 160).any() or (ind_range < 0).any():
            print "n_points too large, defaulting to all"
            ind_range = np.arange(161)
        height = max(order) - min(order)
        p0 = [-height, velocity[i,ind_min], 3.0, max(order)]
        if debug:
            print "order index {0}".format(i)
            print "starting param",p0
        popt, pcov = curve_fit(gauss_function, velocity[i,ind_range], order[ind_range], p0=p0, maxfev=10000)
        order_par[i,:] = popt
        if debug:
            print "solution param",popt
            plt.scatter(velocity[i],order,color='blue')
            plt.scatter(velocity[i,ind_range],order[ind_range],color='red')
            x = np.arange(10000)/100.0 + velocity[i,0]
            plt.plot(x, gauss_function(x,popt[0],popt[1],popt[2],popt[3]))
            plt.ylim(min(ccf[i])-500,max(order))
            plt.show()
    return order_par

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
 
    order_time_par = np.zeros((len(s.files), 73, 4))
    pipeline_rv = np.zeros(len(s.files))
    for i,f in enumerate(s.files):
        velocity, ccf, pipeline_rv[i] = read_ccfs(f)
        order_par = rv_gaussian_fit(velocity, ccf, n_points=20) # chose n_points=20 from the median order RMS plot sent to Hogg on May 13 
        order_time_par[i,:,:] = order_par
        #print "file {0} of {1} finished".format(i+1,len(s.files))
            
    rv = np.nanmedian(order_time_par[:,:-1,1], axis=1) # median of every order RV (excluding the co-added one)
    print "pipeline's RV RMS = {0:.2f} m/s".format(np.std(s.rv)*1.0e3)
    
    
    #try weighted means
    masked_par = np.ma.masked_invalid(order_time_par[:,:-1,:])
    rv_aweight = np.ma.average(masked_par[:,:,1], weights=abs(masked_par[:,:,0]), axis=1)
    rv_a2weight = np.ma.average(masked_par[:,:,1], weights=(masked_par[:,:,0])**2, axis=1)
    print "abs(a) weighted RV RMS = {0:.2f} m/s".format(np.std(rv_aweight)*1.0e3)
    print "a^2 weighted RV RMS = {0:.2f} m/s".format(np.std(rv_a2weight)*1.0e3)
    
    #PCA!
    # subtract off the mean values from every time series:
    a = np.delete(order_time_par, (71,66,57,72), axis=1) # remove co-added CCF and bad orders
    a_rvonly = a[:,:,1] # select RVs    
    a_rvonly -= np.repeat([np.mean(a_rvonly,axis=0)],48,axis=0)  #subtract off the mean value from each order time series
    a_all = np.reshape(a,(48,-1)) # select all Gaussian parameters
    a_all -= np.repeat([np.mean(a_all,axis=0)],48,axis=0)  #subtract off the mean value from each order time series
    
    #do SVD on RVs only:
    u,s,v = np.linalg.svd(a_rvonly, full_matrices=False)
    #plot some eigenvalues:
    plt.scatter(np.arange(48),s**2)
    plt.title('Eigenvalues')
    plt.xlim(-1,50)
    plt.yscale('log')
    plt.ylim(0.00001,0.1)
    plt.savefig('eigenvalues_rv.png')
    plt.clf()
    #plot some eigenvectors:
    plt.plot(v[0,:],color='red')
    plt.plot(v[1,:],color='orange')
    plt.plot(v[2,:],color='green')
    plt.plot(v[3,:],color='blue')
    plt.plot(v[4,:],color='purple')
    plt.xlabel('Order #')
    plt.ylabel(r'$\Delta$ velocity (km/s)')
    plt.title('Eigenvectors')
    plt.savefig('eigenvectors_rv.png')
    plt.clf()
    
    #do SVD on all Gaussian parameters:
    u,s,v = np.linalg.svd(a_all, full_matrices=False)
    #plot some eigenvalues:
    plt.scatter(np.arange(48),s**2)
    plt.title('Eigenvalues')
    plt.xlim(-1,50)
    plt.yscale('log')
    plt.ylim(1.0e6,2.0e14)
    plt.savefig('eigenvalues_all.png')
    plt.clf()
    #plot some eigenvectors:
    plt.plot(v[0,:],color='red')
    plt.plot(v[1,:],color='orange')
    plt.plot(v[2,:],color='green')
    plt.plot(v[3,:],color='blue')
    plt.plot(v[4,:],color='purple')
    plt.xlabel('Order #')
    #plt.ylabel(r'$\Delta$ velocity (km/s)')
    plt.title('Eigenvectors')
    plt.savefig('eigenvectors_all.png')
    plt.clf()

    
    
