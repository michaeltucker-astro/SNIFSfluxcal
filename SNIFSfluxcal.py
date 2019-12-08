import numpy as np
import argparse, glob, os, sys
from astropy.io import fits
from lmfit import Parameters, minimize
from spectres import spectres

import matplotlib.pyplot as plt


from pyExtinction.pyExtinction import AtmosphericExtinction as AtmExt

def main(indir=None, channel=None, plot=True, verb=False):

	channel='BR' if channel is None else channel
	indir = './' if indir is None else indir

	for chan in list(channel):
		if verb: print('Working on channel: ' + chan)
		spex = find1DSpectra(indir=indir, channel=chan)
		
		if len(spex) == 0:
			print('WARNING: No 1D spectra found in dir "%s" for channel "%s" ' % (indir, chan))
			continue
		elif verb: print('\tFound %d 1D spectra in %s' % (len(spex), indir) )

		std_spex = findStdSpex(spex)
		if len(std_spex) == 0:
			print("WARNING: No standard star spectra found! No calibrations will be applied.")
			continue
		elif len(std_spex) < 3:
			print('WARNING: Only %d standard star spectra found, flux calibration will likely be poor quality!' % (len(std_spex)))
		elif verb:
			print('\t%d standard star spectra found.' % (len(std_spex)))

		if verb: print('\tFitting atm throughput and instr resp...')
		fitTransmission(std_spex)
		


class Spectrum:
	def __init__(self, fitsfile):
		self.fname = fitsfile
		if not os.path.exists(self.fname):
			raise IOError('Cannot find input spectrum: %s' % self.fname)
		else:
			self.flux, self.hdr = fits.getdata(self.fname, header=True)

		self.wl = self.hdr['CRVAL1'] + self.hdr['CDELT1']*np.arange(self.flux.size)
		self.varname = os.path.dirname(self.fname)+'/var_'+os.path.basename(self.fname)
		if not os.path.exists(self.varname):
			print('WARNING: Failed to find variance file %s' % self.varname)
			self.err = None
		else:
			self.err = np.sqrt(fits.getdata(self.varname))

		if self.flux.mean() > 1e-5:
			self.flux *= 1e-16
			self.err *= 1e-16

		self.object = self.hdr['OBJECT']
		self.exptime = self.hdr['EXPTIME']
		self.channel=self.hdr['CHANNEL'][0]
		self.am = self.hdr['AIRMASS']
		self.jd = self.hdr['JD']
		self.mjd = self.jd - 2400000.5
		self.date = self.hdr['DATE-OBS']

	def __str__(self):
		outstr = """
### SNIFS spectrum ###
	->File: %s
	->Var: %s
	->Channel: %s
	->Object: %s
	->Exptime: %.1f
	->Airmass: %.2f
	->MJD-OBS: %.5f
	->DATE-OBS: %s
""" % (self.fname, self.varname, self.channel, self.object, self.exptime, 
						self.am, self.mjd, self.date)
		return outstr

	def rebin(self, newwl):
		keep = np.where(
				(newwl > self.wl.min()+5.) &
				(newwl < self.wl.max()-5.)
			)[0]
		scifl, scierr = spectres(newwl[keep], self.wl, self.flux, self.err)
		return scifl, scierr, keep


def findStdSpex(spex):
	std_names, std_RA, std_DEC, std_Vmag, std_stype = np.genfromtxt('./standards.dat', comments='#', dtype=str, unpack=True)
	std_spex = []
	for ii, spec in enumerate(spex):
		if spec.object in list(std_names): std_spex.append(spec)
	return std_spex



def find1DSpectra(indir, channel):
	fnames = glob.glob(indir+'/spec_*_%s.fits' % channel)
	return [Spectrum(ff) for ff in fnames]


def fitTransmission(std_spex):
	am_list, trans_list, terr_list = [],[],[]
	RespFitter(am_list, trans_list, terr_list)

	for spec in std_spex:
		stdwl, stdfl = readStdSpec(name=spec.object)
		try:
			scifl, scierr, stdidx = spec.rebin(stdwl)
		except:
			continue
		trans = scifl / stdfl[stdidx]
		terr = scierr / stdfl[stdidx]
		am_list.append(spec.am)
		trans_list.append(trans)
		terr_list.append(terr)
		plt.plot(stdwl[stdidx], trans, label=spec.object + '-'+str(am_list[-1]))
	plt.legend()
	plt.show()

class RespFitter:
	def __init__(self, wl, am_list, trans_list, terr_list, dichroic=False, atm=True, instr=True, verb=False):
		self.wl = wl
		self.am_list = am_list
		self.trans_list = trans_list
		self.fit_components = {
			'dichroic':dichroic,
			'atm':atm,
			'instr':instr
		}
		self.initialize()

	def initialize(self):
		self.atm_model = AtmExt.ExtinctionModel(lbda=self.wl)
		self.atm_model.setDefaultParams()
		self.P, self.I_O3, self.tau, self.a_dot = self.atm_model.p

	def evaluate(self):
		self.atm_model.setParams([self.P, self.I_O3, self.tau, self.a_dot])
		ext = self.atm_model.extinction()
		trans = 10.0**(-0.4*ext)
		self.updateInstrResp()
		trans *= self.instr_resp
		self.updateDichResp()
		trans *= self.dich_resp
		return trans

	def fit(self):
		params = Parameters()
		params.add('P', value=self.P, vary=False)
		params.add('I_O3', value=self.I_O3, vary=True, bounds=(150., 400.))
		params.add('tau', value=self.tau, vary=True, bounds=(0.001, 0.1))
		params.add('a_dot', value=self.a_dot, vary=True, bounds=(1,5))
		

def readStdSpec(name):
	fname = '/Users/skywalker/Documents/Science/Observing/SNIFS/SNIFSfluxcal/standard_spectra/%s.dat' % name
	if not os.path.exists(fname):
		raise IOError('Cannot find std star spectrum: %s' % fname)
	else:
		wl, mag, binsz = np.genfromtxt(fname, comments='#', dtype=float, unpack=True)
		fnu = 10.0**( (mag+21.1) / -2.5 )
		flam = 29979245800./wl**2.0 * fnu
		return wl, flam



if __name__=='__main__':
	parser = argparse.ArgumentParser(description="Computes atmosperic and instrumental response from a set a SNIFS standard star spectra, then applies corrections to the science spectra.")

	parser.add_argument('--indir', '-d', help='Input directory containing the SNIFS 1D spectra.', default=None, type=str)
	parser.add_argument('--channel', '-c', help='Which channel to process. Default: BR', default='BR', choices=['B','R','BR','RB'])
	parser.add_argument('--plot', '-p', help='Show plots? Default: False', default=False, action='store_true')
	parser.add_argument('--verbose', '-v', help='Verbose output? Default: False', action='store_true')

	args = parser.parse_args()
	main(args.indir, args.channel, args.plot, args.verbose)

