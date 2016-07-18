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
    HIP54287.get_drift(s.files) # fetch instrument drifts
    HIP54287.get_wavepar(s.files) # fetch wavelength solution param
    wavepar = np.delete(HIP54287.wavepar,[71, 66, 57],axis=1)  # remove the orders that have no RVs
    HIP54287.set_param()    
    
    order_time_par = HIP54287.data        
    rv = np.median(order_time_par[:,:,1], axis=1) # median of every order RV
    rv -= HIP54287.drift*1e-3 # apply the drift correction
    print "pipeline's RV RMS = {0:.3f} m/s".format(np.std(s.rv)*1.0e3)
    print "unweighted median RV RMS = {0:.3f} m/s".format(np.std(rv)*1.0e3)

    #try weighted means
    rv_aweight = np.average(order_time_par[:,:,1], weights=abs(order_time_par[:,:,0]), axis=1) - HIP54287.drift*1e-3
    rv_a2weight = np.average(order_time_par[:,:,1], weights=(order_time_par[:,:,0])**2, axis=1) - HIP54287.drift*1e-3
    print "abs(a) weighted mean RV RMS = {0:.3f} m/s".format(np.std(rv_aweight)*1.0e3)
    print "a^2 weighted mean RV RMS = {0:.3f} m/s".format(np.std(rv_a2weight)*1.0e3)
        
    rv_aweightmed = []
    for i in range(len(HIP54287.t)):
        rv_aweightmed.append(weighted_median(order_time_par[i,:,1], ws=abs(order_time_par[i,:,0])))
    #rv_aweightmed = np.apply_along_axis(weighted_median,1,order_time_par[:,:,1])
    rv_aweightmed -= HIP54287.drift*1e-3 # apply the drift correction
    print "abs(a) weighted median RV RMS = {0:.3f} m/s".format(np.std(rv_aweightmed)*1.0e3)
    

    # multiple linear regression with wavelength par
    m = 39  # a high-amplitude CCF order
    v_m = HIP54287.data[:,m,1] # time-series RVs for this order
    N = len(HIP54287.t)
    n_pred = 5 # number of predictors, including intercept term
    A_m = np.ones((N,n_pred)) # design matrix for this order
    A_m[:,1:] = HIP54287.wavepar[:,m,:]
    x_m = np.zeros((N,n_pred))
    l = 0.1  # regularization parameter
    reg_matrix = np.identity(n_pred)  # regularization...
    reg_matrix[0] = 0                 # ... matrix
    v_pred = np.zeros(N) # predicted RVs
    for i in range(N):
        A_noi = np.delete(A_m, i, axis=0)
        v_noi = np.delete(v_m, i)
        x_noi = np.linalg.solve(np.dot(A_noi.T,A_noi)+l*reg_matrix, \
                np.dot(A_noi.T,v_noi)) # best-fit coeffs excluding i-th epoch
        x_m[i,:] = x_noi
        v_pred[i] = np.dot(A_m[i,:], x_noi) # leave-one-out regression prediction for this epoch
        
    
    plt.plot((v_m - v_m[0])*1e3, color='red', label='observed velocities')
    plt.plot((v_pred - v_m[0])*1e3, color='blue', label='predicted from wavelength param')
    plt.xlabel('Epoch #')
    plt.ylabel('RV (m/s)')
    plt.legend()
    plt.savefig('fig/regression_normalorder.png')
    plt.clf()
    
    plt.plot(x_m[:,0]/np.max(abs(x_m[:,0])), label='x_0')
    plt.plot(x_m[:,1]/np.max(abs(x_m[:,1])), label='x_1')
    plt.plot(x_m[:,2]/np.max(abs(x_m[:,2])), label='x_2')
    plt.plot(x_m[:,3]/np.max(abs(x_m[:,3])), label='x_3')
    plt.plot(x_m[:,4]/np.max(abs(x_m[:,4])), label='x_4')
    plt.legend()
    plt.savefig('fig/regression_coeff_normalorder.png')
    plt.clf()
    
    print "RMS of order {0} RVs before regression: {1:.2f} m/s".format(m, np.std(v_m)*1.0e3)
    print "RMS of order {0} residuals to regression: {1:.2f} m/s".format(m, np.std(v_m - v_pred)*1.0e3)
    
    
    m = 3 # a wonky offset order
    v_m = HIP54287.data[:,m,1] # time-series RVs for this order
    N = len(HIP54287.t)
    n_pred = 5 # number of predictors, including intercept term
    A_m = np.ones((N,n_pred)) # design matrix for this order
    A_m[:,1:] = HIP54287.wavepar[:,m,:]
    x_m = np.zeros((N,n_pred))
    l = 0.1  # regularization parameter
    reg_matrix = np.identity(n_pred)  # regularization...
    reg_matrix[0] = 0                 # ... matrix
    v_pred = np.zeros(N) # predicted RVs
    for i in range(N):
        A_noi = np.delete(A_m, i, axis=0)
        v_noi = np.delete(v_m, i)
        x_noi = np.linalg.solve(np.dot(A_noi.T,A_noi)+l*reg_matrix, \
                np.dot(A_noi.T,v_noi)) # best-fit coeffs excluding i-th epoch
        x_m[i,:] = x_noi
        v_pred[i] = np.dot(A_m[i,:], x_noi) # leave-one-out regression prediction for this epoch
         
    plt.plot((v_m - v_m[0])*1e3, color='red', label='observed velocities')
    plt.plot((v_pred - v_m[0])*1e3, color='blue', label='predicted from wavelength param')
    plt.xlabel('Epoch #')
    plt.ylabel('RV (m/s)')
    plt.legend()
    plt.savefig('fig/regression_weirdorder.png')
    plt.clf()
    
    plt.plot(x_m[:,0]/np.max(abs(x_m[:,0])), label='x_0')
    plt.plot(x_m[:,1]/np.max(abs(x_m[:,1])), label='x_1')
    plt.plot(x_m[:,2]/np.max(abs(x_m[:,2])), label='x_2')
    plt.plot(x_m[:,3]/np.max(abs(x_m[:,3])), label='x_3')
    plt.plot(x_m[:,4]/np.max(abs(x_m[:,4])), label='x_4')
    plt.legend()
    plt.savefig('fig/regression_coeff_weirdorder.png')
    plt.clf()
    
    print "RMS of order {0} RVs before regression: {1:.2f} m/s".format(m, np.std(v_m)*1.0e3)
    print "RMS of order {0} residuals to regression: {1:.2f} m/s".format(m, np.std(v_m - v_pred)*1.0e3)
    
