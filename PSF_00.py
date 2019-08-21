# Python3 program to fit a two-dimensional Gaussian function
# to the Point Spread Function of 20 Ophiuchi

# use arithmetic mean of dark images as dark frame for all images
# trim all images around the position of 20 Ophiuchi
from astropy.io import fits
d_list = 1.0*fits.open('/Users/kawaii/Documents/obs/190626/SAO_160118_Rdark_1.0_01.fits')[0].data[:,1060:1160,850:950]
import numpy as np
d = (1/len(d_list))*np.sum(d_list, 0)

# load images and subtract dark frame
r_list = 1.0*fits.open('/Users/kawaii/Documents/obs/190626/SAO_160118_R_1.0_01.fits')[0].data[:,1060:1160,850:950]
for n in range(len(r_list)):
	r_list[n] = r_list[n] - d

# load flat dark images for each filter
# use arithmetic means of flat dark images as flat dark frame
frd_list = [ 1.0*fits.open('/Users/kawaii/Documents/obs/190620/flat_rdark_0.05_0'+n+'.fits')[0].data[0][1060:1160,850:950] for n in ['1', '2', '3', '4', '5'] ]
frd = (1/len(frd_list))*np.sum(frd_list, 0)

# load flat images and subtract flat dark frame
fr_list = [ 1.0*fits.open('/Users/kawaii/Documents/obs/190620/flat_r_0.05_0'+n+'.fits')[0].data[0][1050:1150,850:950] for n in ['1', '2', '3', '4', '5'] ]
fr = (1/len(fr_list))*np.sum(fr_list, 0) - frd

# normalize flat images by setting median pixel to 1.0
fr = fr/np.median(fr)

# adjust images for sensitivity using flat images
for n in range(len(r_list)):
	r_list[n] = r_list[n]/fr
image = (1/len(r_list))*np.sum(r_list, 0)

# PSF fitting
import warnings
from scipy.ndimage import measurements
from astropy.modeling import models, fitting
import math

y, x = np.mgrid[:100, :100]
ind = measurements.center_of_mass(image, labels=None, index=None)
init = models.Gaussian2D(amplitude=np.max(image), x_mean=ind[0], y_mean=ind[1], x_stddev=None, y_stddev=None)
fitter = fitting.LevMarLSQFitter()

with warnings.catch_warnings():
    # Ignore model linearity warning from the fitter
    warnings.simplefilter('ignore')
    fitted = fitter(init, x, y, image)
print("x_fwhm: ",fitted.x_fwhm," pixels")
print("y_fwhm: ",fitted.y_fwhm," pixels")
print("geometric mean of x_fwhm and y_fwhm: ",math.sqrt(fitted.x_fwhm*fitted.y_fwhm)," pixels")
print("estimated fwhm of psf: ",0.14*math.sqrt(fitted.x_fwhm*fitted.y_fwhm)," arcseconds")


# plot
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 2.5))
plt.subplot(1, 4, 1)
plt.imshow(image, origin='lower')
plt.title("Data")
plt.colorbar(orientation='horizontal')
plt.subplot(1, 4, 2)
plt.imshow(init(x, y), origin='lower')
plt.title("Initial Condition")
plt.colorbar(orientation='horizontal')
plt.subplot(1, 4, 3)
plt.imshow(fitted(x, y), origin='lower')
plt.title("Model")
plt.colorbar(orientation='horizontal')
plt.subplot(1, 4, 4)
plt.imshow(image - fitted(x, y), origin='lower')
plt.title("Residual")
plt.colorbar(orientation='horizontal')
plt.show()
