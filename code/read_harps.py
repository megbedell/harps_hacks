import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.gridspec as gridspec
from astropy.io import fits
from scipy.optimize import curve_fit
import pdb
import csv
c = 299792.458 # speed of light in km/s

def read_csv(csv_file):
    '''Read a CSV file with header in as dict
    
    Parameters
    ----------
    csv_file : string
    name of the CSV file
    
    Returns
    -------
    data : dict
    the contents of the CSV file in dictionary form
    '''
    with open(csv_file, 'r' ) as f:
        reader = csv.DictReader(f)
        data = {}
        for row in reader:
            for column, value in row.iteritems():
                data.setdefault(column, []).append(value)
    return data

def read_spec(spec_file):
    '''Read a HARPS 1D spectrum file from the ESO pipeline

    Parameters
    ----------
    spec_file : string
    name of the fits file with the data (s1d format)
    
    Returns
    -------
    wave : np.ndarray
    wavelength (in Angstroms)
    flux : np.ndarray
    flux value
    '''
    sp = fits.open(spec_file)
    header = sp[0].header
    n_wave = header['NAXIS1']
    crval1 = header['CRVAL1']
    cdelt1 = header['CDELT1']
    index = np.arange(n_wave, dtype=np.float64)
    wave = crval1 + index*cdelt1
    flux = sp[0].data
    return wave, flux
    
def read_spec_2d(spec_file, blaze=False, flat=False):
    '''Read a HARPS 2D spectrum file from the ESO pipeline

    Parameters
    ----------
    spec_file : string
    name of the fits file with the data (e2ds format)
    blaze : boolean
    if True, then divide out the blaze function from flux
    flat : boolean
    if True, then divide out the flatfield from flux
    
    Returns
    -------
    wave : np.ndarray (shape n_orders x 4096)
    wavelength (in Angstroms)
    flux : np.ndarray (shape n_orders x 4096)
    flux value 
    '''
    path = spec_file[0:str.rfind(spec_file,'/')+1]
    sp = fits.open(spec_file)
    header = sp[0].header
    flux = sp[0].data
    wave_file = header['HIERARCH ESO DRS CAL TH FILE']
    wave_file = str.replace(wave_file, 'e2ds', 'wave') # just in case of header mistake..
                                                       # ex. HARPS.2013-03-13T09:20:00.346_ccf_M2_A.fits
    try:
        ww = fits.open(path+wave_file)
        wave = ww[0].data
    except:
        print("Wavelength solution file {0} not found!".format(wave_file))
        return
    if blaze:
        blaze_file = header['HIERARCH ESO DRS BLAZE FILE']
        bl = fits.open(path+blaze_file)
        blaze = bl[0].data
        flux /= blaze
    if flat:
        flat_file = header['HIERARCH ESO DRS CAL FLAT FILE']
        fl = fits.open(path+flat_file)
        flat = fl[0].data
        flux /= flat
    return wave, flux
    

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

def read_wavepar(filename):
    '''Parse wavelength solution on a HARPS file from the ESO pipeline

    Parameters
    ----------
    filename : string
    name of the fits file with the data (can be ccf, e2ds, s1d)

    Returns
    -------
    wavepar : np.ndarray
    a 72 x 4 array of wavelength solution parameters
    '''
    sp = fits.open(filename)
    header = sp[0].header
    
    #n_orders = header['NAXIS2']
    n_orders = 72
    wavepar = np.arange(4*n_orders, dtype=np.float).reshape(n_orders,4)
    for i in np.nditer(wavepar, op_flags=['readwrite']):
        i[...] = header['HIERARCH ESO DRS CAL TH COEFF LL{0}'.format(str(int(i)))]
    return wavepar


def read_snr(filename):
    '''Parse SNR from header of a HARPS file from the ESO pipeline

    Parameters
    ----------
    filename : string
    name of the fits file with the data (can be ccf, e2ds, s1d)

    Returns
    -------
    snr : np.ndarray
    SNR values taken near the center of each order
    '''
    sp = fits.open(filename)
    header = sp[0].header
    
    #n_orders = header['NAXIS2']
    n_orders = 72
    snr = np.arange(n_orders, dtype=np.float)
    for i in np.nditer(snr, op_flags=['readwrite']):
        i[...] = header['HIERARCH ESO DRS SPE EXT SN{0}'.format(str(int(i)))]
    return snr


def read_drift(filename):
    '''Parse simultaneous reference drift from header of a HARPS file from the ESO pipeline

    Parameters
    ----------
    filename : string
    name of the fits file with the data (must be ccf)

    Returns
    -------
    drift : the value of RV drift to be subtracted from the CCF in m/s
    '''
    sp = fits.open(filename)
    header = sp[0].header
    drift = header['HIERARCH ESO DRS DRIFT RV USED']
    return drift  # to do: fix so this works for data not in simultaneous ref mode
       


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
    

def rv_gaussian_fit(velocity, ccf, n_points=20, mask_inner=0, debug=False, all=False):
    '''Read in the pipeline CCF data product and return Gaussian-fitted RV for each order
    
    Parameters
    ----------
    velocity : np.ndarray
    velocity (in km/s)
    ccf : np.ndarray
    ccf value
    n_points : int
    total number of points on either side of the minimum to fit, must be >= 2
    mask_inner : int
    number of points to ignore on either side of the minimum when fitting (including minimum itself)
    (if 1, ignore the minimum point only)
    debug : boolean
    if True then include print-outs and show a plot
    all : boolean
    if True then include the three non-functional orders and the co-added "order"

    Returns
    -------
    order_par : np.ndarray
    best-fit Gaussian parameters for each order: (amplitude, mean, sigma, offset)
    '''
    if (n_points < 1):
        print("Cannot fit a Gaussian to < 4 points! Try n_points = 2")
        return None
    order_par = np.zeros((ccf.shape[0],4))
    for i,order in enumerate(ccf):
        if i in [71, 66, 57]:  # disregard the bad orders
            order_par[i,:] = np.nan
            continue
        ind_min = np.argmin(order)
        ind_range = np.arange(n_points*2+1) + ind_min - n_points
        if (ind_range > 160).any() or (ind_range < 0).any():
            print("n_points too large, defaulting to all")
            ind_range = np.arange(161)
        ind_range = np.delete(ind_range, np.where(np.abs(ind_range - ind_min) < mask_inner))
        height = max(order) - min(order)
        p0 = [-height, velocity[i,ind_min], 3.0, max(order)]
        if debug:
            print("order index {0}".format(i))
            print("starting param",p0)
        popt, pcov = curve_fit(gauss_function, velocity[i,ind_range], order[ind_range], p0=p0, maxfev=10000)
        order_par[i,:] = popt
        if debug:
            print("solution param",popt)
            plt.scatter(velocity[i],order,color='blue')
            plt.scatter(velocity[i,ind_range],order[ind_range],color='red')
            x = np.arange(10000)/100.0 + velocity[i,0]
            plt.plot(x, gauss_function(x,popt[0],popt[1],popt[2],popt[3]))
            plt.ylim(min(ccf[i])-500,max(order))
            plt.show()
    
    if not all:
        order_par = np.delete(order_par, [57,66,71,72], axis=0)
    return order_par

def rv_oneline_fit(wave, spec, wave_c, debug=False):
    '''Read in the pipeline CCF data product and return Gaussian-fitted RV for each order
    
    Parameters
    ----------
    wave : np.ndarray
    wavelengths of the spectrum
    spec : np.ndarray
    spectrum
    wave_c : float
    starting guess of the line's central wavelength
    debug : boolean
    if True then include print-outs and show a plot
    
    Returns
    -------
    rv : np.float64
    RV shift of the line
    '''
    
    ind = np.where(np.logical_and(wave >= wave_c - 1.5, wave <= wave_c + 1.5)) # selected region of spectrum
    wave = wave[ind] # trim to smaller size for fitting
    spec = spec[ind]
    # fit a Gaussian:
    height = max(spec) - min(spec)
    p0 = [-height, wave_c, 0.3, max(spec)]
    popt, pcov = curve_fit(gauss_function, wave, spec, p0=p0, maxfev=10000)
    wave_obs = popt[1]
    # get Doppler shift:
    c = 299792.458 # km/s
    rv = (wave_obs - wave_c)/wave_c * c
    if debug:
        print("solution param",popt)
        plt.scatter(wave,spec)
        x = np.arange(10000)/3000.0 + wave[0]
        plt.plot(x, gauss_function(x,popt[0],popt[1],popt[2],popt[3]))
        plt.show()
    return rv