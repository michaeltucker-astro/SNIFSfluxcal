import numpy as np
import argparse, glob
from astropy.io import fits

import matplotlib.pyplot as plt


from pyExtinction.pyExtinction import AtmosphericExtinction as AtmExt

def main(indir=None, channel=None, plot=True):

	channel='BR' if channel is None else channel
	indir = './' if indir is None else indir

	for chan in list(channel):
		if verb: print('Working on channel: ' + chan)
		spex = find1DSpectra(indir=indir, channel=chan)
		
		if len(spex) == 0:
			print('WARNING: No 1D spectra found in dir "%s" for channel "%s" ' % (indir, chan))
			continue
		elif verb: print('\tFound %d 1D spectra in %s' % (len(spex), indir) )





class Spectrum:
	def __init__(self, fitsfile):
		self.fname = fitsfile
		if not os.path.exists(self.fname):
			raise IOError('Cannot find input spectrum: %s' % self.fname)
		else:
			self.flux, self.hdr = fits.getdata(self.fname, header=True)

		self.wl = hdr['CRVAL1'] + hdr['CDELT1']*np.arange(self.fl.size)
		self.varname = os.path.dirname(self.fname)+'/var_'+os.path.basename(self.fname)
		if not os.path.exists(self.varname):
			print('WARNING: Failed to find variance file %s' % self.varname)
			self.err = None
		else:
			self.err = np.sqrt(fits.getdata(self.varname))
		self.channel=self.hdr['CHANNEL']
		self.am = self.hdr['AIRMASS']

	def __str__(self):
		outstr = """
				### SNIFS spectrum ###
				->File: %s
				->Var: %s
				->Channel: %s
				->Object: %s
				->Exptime: %.1f
				->Airmass: %.1f
				->MJD-OBS: %.5f
				->DATE-OBS: %s
				""" % (self.fname, self.varname, self.channel, self.object, self.exptime, 
						self.am, self.mjd, self.date)
		return outstr





def find1DSpectra(indir, channel):
	fnames = glob.glob(indir+'/spec_*_%s.fits' % channel)
	return fnames



if __name__=='__main__':
	parser = argparse.ArgumentParser(description="Computes atmosperic and instrumental response from a set a SNIFS standard star spectra, then applies corrections to the science spectra.")

	parser.add_argument('--indir', '-d', help='Input directory containing the SNIFS 1D spectra.')