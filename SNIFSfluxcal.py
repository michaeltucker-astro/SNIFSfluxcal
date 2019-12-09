import numpy as np
import glob, os, sys
from astropy.io import fits
from lmfit import Parameters, minimize
from spectres import spectres
import ipdb
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

	for spec in std_spex:
		stdwl, stdfl = readStdSpec(name=spec.object)
		try:
			scifl, scierr, stdidx = spec.rebin(stdwl)
		except:
			continue
		trans = scifl / stdfl[stdidx]
		trans_interp = np.interp(spec.wl, stdwl[stdidx], trans)
		terr = scierr / stdfl[stdidx]
		terr_interp = np.interp(spec.wl, stdwl[stdidx], terr)
		am_list.append(spec.am)
		trans_list.append(trans_interp)
		terr_list.append(terr_interp)

	fit = RespFitter(std_spex[0].wl, am_list, trans_list, terr_list)
	ipdb.set_trace()


class RespFitter:
	def __init__(self, wl, am_list, trans_list, terr_list, dichroic=False, atm=True, instr=True, verb=False):
		self.wl = wl
		self.am_list = am_list
		self.trans_list = trans_list
		self.terr_list = terr_list
		self.fit_components = {
			'dichroic':dichroic,
			'atm':atm,
			'instr':instr
		}

		self.default_params = {
			'I_O3':[260.0, 50.0],
			'tau':[0.007, 0.004],
			'a_dot':[1.0, 3.0]
		}
		self.initialize()

	def initialize(self):
		self.atm_model = AtmExt.ExtinctionModel(lbda=self.wl)
		self.atm_model.setDefaultParams()
		self.P, self.I_O3, self.tau, self.a_dot = self.atm_model.p

	def compute_atm_trans(self, am):
		ext = self.atm_model.extinction()[0] * am
		trans = 10.0**(-0.4*ext)
		return trans

	def compute_residuals(self):
		residuals = []
		dich_trans = self.compute_dich_trans()
		instr_trans = self.compute_instr_trans()
		for am, std_trans, std_terr in zip(self.am_list, self.trans_list, self.terr_list):
			atm_trans = self.compute_atm_trans(am=am)
			trans = atm_trans * dich_trans * instr_trans
			resid = (trans - std_trans)**2.0
			residuals.append(resid)
		return np.stack(residuals)

	def compute_instr_trans(self):
		return 0.01*np.ones_like(self.wl)

	def compute_dich_trans(self):
		return np.ones_like(self.wl)

	def compute_priors_penalty(self):
		p0, dp0 = self.default_params['I_O3']
		term1 = ((p0-self.I_O3)/dp0)**2.0
		p0, dp0 = self.default_params['a_dot']
		term2 = ((p0-self.a_dot)/dp0)**2.0
		p0, dp0 = self.default_params['tau']
		lp0 = np.log(self.tau/p0)
		ldp0 = np.log(dp0)
		term3 = (lp0/ldp0)**2.0
		return term1*term2*term3

	def penalty_fcn(self, params):
		pars = params.valuesdict()
		self.I_O3 = pars['I_O3']
		self.tau = pars['tau']
		self.a_dot = pars['a_dot']
		self.P = pars['P']

		self.atm_model.setParams([self.P, self.I_O3, self.tau, self.a_dot])

		chi2 = self.compute_residuals()
		psi2 = self.compute_priors_penalty()
		total_chi2 = np.sqrt( chi2 + psi2 )
		return total_chi2

	def fit(self):
		params = Parameters()
		params.add('P', value=self.P, vary=False)
		params.add('I_O3', value=self.I_O3, vary=True, min=150., max=400.)
		params.add('tau', value=self.tau, vary=True, min=0.001, max=0.1)
		params.add('a_dot', value=self.a_dot, vary=True, min=1,max=5)

		self.result = minimize(self.penalty_fcn, params=params)





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
	import argparse
	parser = argparse.ArgumentParser(description="Computes atmosperic and instrumental response from a set a SNIFS standard star spectra, then applies corrections to the science spectra.")

	parser.add_argument('--indir', '-d', help='Input directory containing the SNIFS 1D spectra.', default=None, type=str)
	parser.add_argument('--channel', '-c', help='Which channel to process. Default: BR', default='BR', choices=['B','R','BR','RB'])
	parser.add_argument('--plot', '-p', help='Show plots? Default: False', default=False, action='store_true')
	parser.add_argument('--verbose', '-v', help='Verbose output? Default: False', action='store_true')

	args = parser.parse_args()
	main(args.indir, args.channel, args.plot, args.verbose)

