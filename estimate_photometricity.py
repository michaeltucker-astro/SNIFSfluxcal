import numpy as np
from astropy.io import fits
import os, sys
from scipy.optimize import curve_fit
from astropy.modeling.models import Gaussian2D
from astropy.stats import sigma_clip

def estimate_photometricity(guider_file, binsz=5, stats=True, plot=False, verb=False):
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

	print('\n'+'#'*15)
	print('Analyzing guider file: %s' % guider_file)

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
			_fwhm, _dfwhm, _flux, _dflux = 2.0, 5.0, 0.0, 1e5
		fwhm.append(_fwhm)
		fwhm_err.append(_dfwhm)
		flux.append(_flux)
		flux_err.append(_dflux)
		n += binsz

	FWHM, dFWHM, Flux, dFlux, is_photometric = measure_photometricity(fwhm, fwhm_err, flux, flux_err, stats=stats)

	if stats:
		print('FWHM: %.2f +/- %.2f"' % (FWHM, dFWHM))
		print('Flux: %.1f +/- %.1f counts/frame' % (Flux, dFlux))
		print('Photometric: %r' % is_photometric)



	if plot and is_photometric is not None:
		if verb: print('Plotting results...')
		import matplotlib.pyplot as plt

		x = np.arange(len(fwhm))+1.
		plt.errorbar(x, fwhm, yerr=fwhm_err, color='r', marker='.', ls=':')
		plt.ylabel('FWHM [arcsec]', color='r', weight='heavy', fontsize=15)
		plt.xlabel('Frame #', fontsize=15)
		plt.yticks(color='r')
		ax2 = plt.gca().twinx()
		ax2.errorbar(x, flux, yerr=flux_err, color='b', marker='.', ls='--')
		plt.ylabel('Flux [counts]', color='b', weight='heavy', fontsize=15)
		plt.yticks(color='b')
		plt.tight_layout()
		plt.show()

	return {'FWHM':[FWHM, dFWHM], 'flux':[Flux, dFlux], 'photometric':is_photometric}

def measure_photometricity(fwhm, fwhm_err, flux, flux_err, stats=False):
	fwhm = np.array(fwhm)
	fwhm_err = np.array(fwhm_err)
	flux = np.array(flux)
	flux_err = np.array(flux_err)

	if np.isfinite(fwhm).sum() < len(fwhm):
		return 0,0,0,0,False


	is_photometric = True
	fwhm_mean, fwhm_median, fwhm_std, _phot = compute_stats(fwhm, fwhm_err)
	flux_mean, flux_median, flux_std, _phot = compute_stats(flux, flux_err)
	is_photometric = is_photometric and _phot
	if len(fwhm) < 10: is_photometric = None
	elif len(fwhm[fwhm>=2.0]) > 0: is_photometric = False
	elif fwhm_std > 0.1: is_photometric = False
	elif fwhm_mean < 0.75: is_photometric = True

	if not is_photometric:
		return fwhm_mean, fwhm_std, flux_mean, flux_std, is_photometric

	x = np.arange(fwhm.size)+1.
	(m, b), cov = np.polyfit(x, flux, 1, cov=True, w=flux_err**-2.)
	dm, db = np.sqrt(np.diag(cov))
	if stats:
		print('Linear fit: m=(%.2g +/- %.2g); b=(%.2g +/- %.2g)' % (m, dm, b, db))

	if abs(m/dm) > 3.: is_photometric=False
	return fwhm_mean, fwhm_std, flux_mean, flux_std, is_photometric

def compute_stats(arr, err):
	mean = np.average(arr, weights=err**-1.)
	median = np.median(arr)
	std = np.std(arr, ddof=1)

	if abs(mean - median)/std > 0.5:
		return mean, median, std, False

	mask = sigma_clip(arr, maxiters=None, sigma=4).mask
	if mask.sum() > 0:
		_phot = False
	else:
		_phot=True

	return mean, median, std, _phot




def fit2Dgauss(image):
	def _model(xy, amp, x_mean, y_mean, x_std, y_std):
		gauss = Gaussian2D(amplitude=amp, x_mean=x_mean, y_mean=y_mean, x_stddev=x_std, y_stddev=y_std)
		fit = gauss(*xy)
		return fit.ravel()

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
	fwhm_err = 2.35 * np.sqrt( ((perr[3:5])**2.0).sum() ) / 2.0 * 0.24

	return fwhm, fwhm_err, flux, flux_err

def readGuiderFile(fname):
	data, hdr = fits.getdata(fname, header=True)
	idx = int(list(str(hdr['GSREGION'])).index('2')/2)
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