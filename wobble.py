import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit


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
    fig.subplots_adjust(bottom=0.025, left=0.025, top = 0.975, right=0.975)
    
    for i in np.arange(72):
        # plot the CCF for this order
        plt.subplot(18,4,i+1)
        plt.plot(velocity[i,:],ccf[i,:])
        plt.xticks([-10.0,0.0,10.0,20.0]), plt.yticks([])
        plt.tick_params(labelsize=7)
        # find the central value and mark it
    
    fig.savefig(file_out)



if __name__ == "__main__":
    
    file_in = "/Users/mbedell/Documents/Research/HARPSTwins/Data/Reduced/2011-10-11/HARPS.2011-10-12T07:19:35.401_ccf_G2_A.fits"
    velocity, ccf = read_ccfs(file_in)
    plot_ccfs(velocity, ccf, file_out='all_ccfs.png')
