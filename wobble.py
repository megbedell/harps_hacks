import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


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
    normalized ccf value
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
    normalized ccf value
    file_out : string (optional keyword)
    name for the output plot; default- 'all_ccfs.png'

    Returns
    -------
    none
    '''



if __name__ == "__main__":
    
    file_in = "/Users/mbedell/Documents/Research/HARPSTwins/Data/Reduced/2011-10-11/HARPS.2011-10-12T07:19:35.401_ccf_G2_A.fits"
    velocity, ccf = read_ccfs(file_in)
    plot_ccfs(velocity, ccf, file_out='all_ccfs.png')
