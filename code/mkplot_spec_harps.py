import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.gridspec as gridspec
import matplotlib
import numpy as np
from astropy.io import fits
c = 299792.458 # speed of light (km/s)

#import some data
sp = fits.open('/Users/mbedell/Documents/Research/HARPSTwins/Data/Reduced/2013-03-24/HARPS.2013-03-24T23:38:48.290_s1d_A.fits')
header = sp[0].header
naxis1 = header['NAXIS1']
crval1 = header['CRVAL1']
cdelt1 = header['CDELT1']
wave1 = crval1 + np.arange(naxis1)*cdelt1
flux1 = sp[0].data
sp = fits.open('/Users/mbedell/Documents/Research/HARPSTwins/Data/Reduced/2013-03-24/HARPS.2013-03-24T23:38:48.290_ccf_G2_A.fits')
header = sp[0].header
rv1 = header['HIERARCH ESO DRS CCF RV']
wave1 /= 1.0 + rv1/c

sp = fits.open('/Users/mbedell/Documents/Research/HARPSTwins/Data/Reduced/2013-03-26/HARPS.2013-03-26T23:49:39.952_s1d_A.fits')
header = sp[0].header
naxis1 = header['NAXIS1']
crval1 = header['CRVAL1']
cdelt1 = header['CDELT1']
wave2 = crval1 + np.arange(naxis1)*cdelt1
flux2 = sp[0].data
sp = fits.open('/Users/mbedell/Documents/Research/HARPSTwins/Data/Reduced/2013-03-26/HARPS.2013-03-26T23:49:39.952_ccf_G2_A.fits')
header = sp[0].header
rv2 = header['HIERARCH ESO DRS CCF RV']
wave2 /= 1.0 + rv2/c

sp = fits.open('/Users/mbedell/Documents/Research/HARPSTwins/Data/Reduced/2013-03-29/HARPS.2013-03-29T23:33:06.408_s1d_A.fits')
header = sp[0].header
naxis1 = header['NAXIS1']
crval1 = header['CRVAL1']
cdelt1 = header['CDELT1']
wave3 = crval1 + np.arange(naxis1)*cdelt1
flux3 = sp[0].data
sp = fits.open('/Users/mbedell/Documents/Research/HARPSTwins/Data/Reduced/2013-03-29/HARPS.2013-03-29T23:33:06.408_ccf_G2_A.fits')
header = sp[0].header
rv3 = header['HIERARCH ESO DRS CCF RV']
wave3 /= 1.0 + rv3/c


#set up the figure
#matplotlib.rcParams['xtick.labelsize'] = 20
#matplotlib.rcParams['ytick.labelsize'] = 20

fig = plt.figure()
gs = gridspec.GridSpec(2,1,height_ratios=[4,1],hspace=0.1)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax2.ticklabel_format(useOffset=False)
majorLocator = MultipleLocator(1)
minorLocator = MultipleLocator(0.1)
ax1.xaxis.set_minor_locator(minorLocator)
ax1.xaxis.set_major_locator(majorLocator)
ax2.xaxis.set_minor_locator(minorLocator)
ax2.xaxis.set_major_locator(majorLocator)
majorLocator = MultipleLocator(0.05)
ax2.yaxis.set_major_locator(majorLocator)
plt.setp(ax1.get_xticklabels(), visible=False)


c1 = 'black'
c2 = '#003399'
c3 = '#CC0033'

#plot spectra
flux1 /= 24500.0
flux2 /= 34500.0
flux3 /= 31000.0
ax1.plot(wave1,flux1,label=r'$\Delta$RV = 0 m/s',color=c1,lw=1.5)
ax1.plot(wave2,flux2,label=r'$\Delta$RV = +14 m/s',color=c2,lw=1.5)
ax1.plot(wave3,flux3,label=r'$\Delta$RV = -13 m/s',color=c3,lw=1.5)
ax1.text(6247.55,0.6,'Fe II',ha='center',size=24)
ax1.text(6246.3,0.35,'Fe I',ha='center',size=24)
ax1.text(6245.6,0.67,'Sc II',ha='center',size=24)
ax1.text(6244.45,0.65,'Si I',ha='center',size=24)

ax1.set_xlim([6244.0,6248.0])
#ax1.set_xlim([5135.0, 5139.0])
ax1.set_ylim([0.31,1.05])
ax1.set_ylabel(r'Normalized Flux',size=28)
ax1.legend(loc='lower left',prop={'size':24})

Sun_resids = flux2 - np.interp(wave2, wave1, flux1)
HD1178_resids = flux3 - np.interp(wave3, wave1, flux1)

ax2.plot(wave1,np.zeros_like(wave1),color=c1)
ax2.plot(wave2,Sun_resids,color=c2)
ax2.plot(wave3,HD1178_resids,color=c3)
ax2.set_ylim([-0.05,0.05])
ax2.set_ylabel(r'Diff.',size=28)
ax2.set_xlabel(r'Wavelength ($\AA$)',size=28)

plt.savefig('spec.png')
#plt.show()