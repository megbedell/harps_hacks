import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from astropy.io import fits
from scipy.optimize import curve_fit
import pdb


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
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + offset

def plot_ccfs(velocity, ccf, pipeline_rv, file_out='all_ccfs.png', calc_sum=False):
    '''Make a multipanel plot of all CCFs, order-by-order

    Parameters
    ----------
    velocity : np.ndarray
    velocity (in km/s)
    ccf : np.ndarray
    ccf value
    pipeline_rv : float
    the HARPS-determined RV from the FITS header (km/s)
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



if __name__ == "__main__":
    
    data_dir = "/Users/mbedell/Documents/Research/HARPSTwins/Data/Reduced/"
    #file_in = "2011-10-11/HARPS.2011-10-12T07:19:35.401_ccf_G2_A.fits"
    file_in = "2015-01-05/HARPS.2015-01-06T06:15:11.910_ccf_G2_A.fits"
    velocity, ccf, rv = read_ccfs(data_dir+file_in)
    plot_ccfs(velocity, ccf, rv, file_out='all_ccfs.png')
