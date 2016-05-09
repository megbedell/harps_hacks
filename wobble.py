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

    return velocity, ccf

def gauss_function(x, a, x0, sigma, offset):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + offset

def plot_ccfs(velocity, ccf, file_out='all_ccfs.png'):
    '''Make a multipanel plot of all CCFs, order-by-order

    Parameters
    ----------
    velocity : np.ndarray
    velocity (in km/s)
    ccf : np.ndarray
    ccf value
    file_out : string (optional keyword)
    name for the output plot; default- 'all_ccfs.png'

    Returns
    -------
    none
    '''
    fig = plt.figure(figsize=(5,20))
    fig.subplots_adjust(bottom=0.025, left=0.025, top = 0.975, right=0.975, hspace=1.0)
    majorLocator   = MultipleLocator(10)
    minorLocator   = MultipleLocator(5)
    j=1
    for i in np.arange(72):
        # skip the bad orders
        if i in [71, 66, 57]:
            continue
        # plot the CCF for this order
        ax = fig.add_subplot(18,4,j)
        j += 1
        ax.plot(velocity[i],ccf[i])
        ax.yaxis.set_ticks([])
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.tick_params(labelsize=7, length=3)
        # find the central value and mark it        
        popt, pcov = curve_fit(gauss_function, velocity[i], ccf[i], p0 = [(min(ccf[i]) - max(ccf[i])), 7.0, 3.0, max(ccf[i])])
        rv = popt[1]
        ax.axvline(rv, color='black', linestyle='--')
        ax.set_title('Order {0}\nRV {1:.2f} km/s'.format(i+1, rv), fontsize=7)

        
    fig.savefig(file_out, dpi=300)



if __name__ == "__main__":
    
    file_in = "/Users/mbedell/Documents/Research/HARPSTwins/Data/Reduced/2011-10-11/HARPS.2011-10-12T07:19:35.401_ccf_G2_A.fits"
    velocity, ccf = read_ccfs(file_in)
    plot_ccfs(velocity, ccf, file_out='all_ccfs.png')
