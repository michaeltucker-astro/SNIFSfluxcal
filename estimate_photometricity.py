import numpy as np
from astropy.io import fits
import os, sys
from scipy.optimize import curve_fit
from astropy.modeling.models import Gaussian2D

import matplotlib.pyplot as plt

def estimate_photometricity(guider_file, binsz=5, plot=False, verb=False):
	if not os.path.exists(guider_file):
		raise IOError('Cannot find guide video file: %s' % guider_file)
	elif not guider_file.endswith('vid.fits'):
		raise ValueError('Expected guider video file (*_vid.fits)!')
	else:
		try:
			frames = readGuiderFile(guider_file)
		except IOError:
			print("ERROR: Cannot read FITS file %s" % guider_file)
			return
	if verb:
		print('Number of frames: %d' % frames.shape[0])

	n = 0
	fwhm = []
	fwhm_err = []
	flux = []
	flux_err = []
	while n < frames.shape[0]:
		slices = frames[n:n+binsz,:,:]
		coadd = slices.sum(axis=0)/slices.shape[0]
		try:
			_fwhm, _dfwhm, _flux, _dflux = fit2Dgauss(coadd)
		except:
			print('ERROR: Failed to fit guider image at n=%d:%d, skipping...' % (n, n+binsz))
			_fwhm, _dfwhm, _flux, _dflux = np.nan, np.nan, np.nan, np.nan
		print(_fwhm, _dfwhm, _flux, _dflux)
		fwhm.append(_fwhm)
		fwhm_err.append(_dfwhm)
		flux.append(_flux)
		flux_err.append(_dflux)
		n += binsz

	if plot:
		if verb: print('Plotting results...')
		import matplotlib.pyplot as plt

		"""
		fig, ax = plt.subplots(111)
		ax[0].errorbar(x, fwhm, yerr=fwhm_err, color='r', marker='.', ls=':')
		ax[0].set_ylabel('FWHM', fontsize=15, color='r')
		ax[0].set_xlabel('Coadd #')
#		ax2 = ax[0].twinx()
#		ax2.errorbar(x, flux, yerr=flux_err, color='b', marker='.', ls='--')
#		ax2.set_ylabel('Flux', fontsize=15, color='b')
		"""
		x = np.arange(len(fwhm))+1.
		plt.errorbar(x, fwhm, yerr=fwhm_err, color='r', marker='.', ls=':')
		ax2 = plt.gca().twinx()
		ax2.errorbar(x, flux, yerr=flux_err, color='b', marker='.', ls='--')
		plt.tight_layout()
		plt.show()


def fit2Dgauss(image):
	def _model(xy, amp, x_mean, y_mean, x_std, y_std):
		gauss = Gaussian2D(amplitude=amp, x_mean=x_mean, y_mean=y_mean, x_stddev=x_std, y_stddev=y_std)
		fit = gauss(*xy)
		return fit.ravel()


	plt.imshow(image, origin='lower')
	plt.show()
	bounds = (
		[np.median(image), 0.0, 0.0, 0.0, 0.0],
		[image.sum(), image.shape[0], image.shape[0], image.shape[0], image.shape[0]]
		)
	flattened = image.ravel() - np.median(image)
	errs = np.sqrt(image).ravel()
	xy = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
	p0 = [image.max() - 0.5*image.mean(), image.shape[1]/2., image.shape[0]/2., image.shape[1]/10., image.shape[0]/10.]
	pfit, pcov = curve_fit(_model, xy, flattened, p0=p0, bounds=bounds, sigma=errs**-1., absolute_sigma=False)
	perr = np.sqrt(np.diag(pcov))

	flux = np.sqrt(2.0*np.pi)*pfit[0]*np.mean(pfit[3:5])
	flux_err = flux * np.sqrt((perr[3]/pfit[3])**2.0 + (perr[4]/pfit[4])**2.0)

	fwhm = 2.35 * np.mean(pfit[3:5]) * 0.2
	fwhm_err = 2.35 * np.sqrt( ((perr[3:5])**2.0).sum() ) / 2.0 * 0.2

	return fwhm, fwhm_err, flux, flux_err

def readGuiderFile(fname):
	data = fits.getdata(fname)
	coadd = data.sum(axis=0)
	plt.imshow(coadd, origin='lower')
	plt.show()
	std = []
	for i in range(4):
		slice = data[:, 32*i:32*(i+1), :]
		plt.imshow(slice, origin='lower')
		plt.show()
	idx = np.array(std).argmax()
	print('Guide star detected on chip: %d' % (idx+1))
	data = data[:, 1:-1, idx*32+1:(idx+1)*32-1].copy()
	return data

if __name__=='__main__':
	import argparse
	parser = argparse.ArgumentParser(description='Uses a set of guider frames to estimate the photometricity of a given exposure', usage='estimate_photometricity [options] guider_file1 [guider_file2 guider_file3 ...]')
	parser.add_argument('filename', help='File(s) to process.', nargs='+')
	parser.add_argument('--plot', '-p', default=False, help='Plot results? Default: False', action='store_true')
	parser.add_argument('--verbose', '-v', help='Verbose output? Default: False', default=False, action='store_true')

	args = parser.parse_args()
	print(args.filename)
	for ff in args.filename:
		estimate_photometricity(ff, plot=args.plot, verb=args.verbose)