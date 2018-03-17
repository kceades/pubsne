"""
Project: pubsne
Authors:
	kceades
"""

# import from Sam's class file
import IDRTools

# import from files I wrote
import tools

# imports from standard packages
import numpy as np
from scipy.interpolate import interp1d
import scipy.signal
import sncosmo
import warnings
from astropy.cosmology import Planck15 as cosmo
import os
from os import walk
import csv
import pickle
import extinction



class Parameters(object):
	"""
	Class to store all the Parameters for the supernova files, including
	how to import and process them. Here is where changes should be made to,
	for example, the standard wavelength bin that they are all put in.
	"""
	def __init__(self):
		""" Constructor """
		# the wavelength bin that all the read in supernovas is re-normalized to
		self.re_start = 3800
		self.re_end = 8000
		self.re_step = 5
		self.wvs = np.arange(self.re_start,self.re_end+self.re_step \
			,self.re_step).astype(float)
		self.num_points = len(self.wvs)
		# the phase list that we are looking at supernova over
		self.phases = np.arange(-10,45,5)
		self.num_phases = len(self.phases)
		# the cut (one-sided) in which to bin the novas within
		self.day_cut = 2.5
		# sources
		self.sources = ['All','Factory','Public','Bsnip']
		# types (observed from the scraper module)
		self.types = ['Ia','Ia Pec','Ia-HV','Ia-SC','Ia-91T','Ia-91bg','I'\
			,'Ia-02cx','LGRB','Ia-09dc','II','II P','Ib','IIn','Ia CSM','Star'\
			,'Ic?','Ic','Ia-02ic-like','Iax[02cx-like]','Ia?','Ia-CSM/IIn'\
			,'Ia-p','Ia-91Tlike','Ia Pec?','Ia/91T']

		# directory
		self.base_dir = os.getcwd()

		# parameters for manipulating spectra
		self.butter_order = 5
		self.butter_cut = 0.1

		# parameter for r_v
		self.r_v = 3.1 # typical value for the Milky Way

		# overall flux to scale to
		self.total_flux = 1*self.num_points
		self.b_flux = 10000
		self.max_flux = 2

		# maximum error acceptable (relative to flux)
		self.max_error = 0.5

		# where the BSNIP spectra are stored
		self.bsnip_path = '/home/kceades/Documents/research/BSNIPI_spectra/'
		# where the public metadata is stored (from the scraper module)
		self.pubmeta_file = '/home/kceades/Documents/research/pubmeta.p'



class SnSet(object):
	"""
	Creates a sort of dataset containing all the supernova from factory and
	public data, prioritizing them in that order (in terms of non-repeating)
	"""
	def __init__(self,pop_source='Public',normalize=True):
		"""
		Constructor

		:pop_source: (str) from Parameters.sources
		:normalize: (bool) whether to normalize the data or not
		"""
		self.pars = Parameters()
		self.keys = {}
		self.data = {}
		self.odd_data = {}

		if pop_source in ['Factory','All']:
			self.pop_factory()
		if pop_source in ['Public','All']:
			self.pop_public()
		if pop_source in ['Bsnip','All']:
			self.pop_bsnip()

		if normalize:
			self.normalize()

		self.phase_novas = {phase:[] for phase in self.pars.phases}
		self.pop_phases(pop_source)

	def deredden(self,wvs,flux,ebv):
		"""
		Removes the extinction from the supernova based on the quoted ebv value

		:wvs: (float array) wavelengths
		:flux: (float array) flux values
		:ebv: (float) the quoted color excess pulled from the scraper

		:returns: (float array) corrected flux values
		"""
		wvs = np.array(wvs)
		dust = extinction.ccm89(wvs,self.pars.r_v*ebv,self.pars.r_v)
		exp_dust = np.power(10,np.multiply(-0.4,dust))
		return np.divide(flux,exp_dust)

	def fill_error(self,flux,errs):
		"""
		Fills the error bins

		:flux: (float array) flux values
		:errs: (float array) error values, some of which are None elements

		:returns: (float array) filled in error array
		"""
		b,a = scipy.signal.butter(self.pars.butter_order,self.pars.butter_cut)
		smoothed = scipy.signal.filtfilt(b,a,flux)
		length = len(flux)
		noise = tools.fit_rms(smoothed,flux)
		return [noise for x in range(length)]

	def normalize(self,mode='b'):
		"""
		Normalizes all the supernova flux data (explained in the Spectrum class
		under the normalize method).

		:mode: (str) 'total','b', or 'max'
		"""
		for sn in self.data:
			for spec in self.data[sn]:
				spec.normalize(mode)

	def pop_bsnip(self):
		"""
		Goes through the BSNIP spectra that I have stored in a local
		folder as well as the manifest file for the redshifts. Note that
		these spectra are all nominally at max light.
		"""
		names = []
		for (dirpath,dirnames,filenames) in walk(self.pars.bsnip_path):
			for nfile in filenames:
				if nfile[0] == 's':
					names.append(nfile)
		bsnipreds = {}
		redfile = open(self.pars.bsnip_path + 'manifest.txt')
		reddata = csv.reader(redfile,delimiter='\n')
		for row in reddata:
			redlist = row[0].split()
			bsnipreds[tools.remove_dash('SN' + redlist[0][17:23])] \
				= float(redlist[1])
		redfile.close()
		for file in names:
			name = 'SN' + tools.remove_dash(file[2:8])
			if name in self.keys:
				continue
			self.data[name] = []
			cfile = open(self.pars.bsnip_path + file)
			fdata = csv.reader(cfile,delimiter='\n')
			cdict = {'w':[],'f':[],'v':[]}
			for row in fdata:
				datalist = row[0].split()
				clength = len(datalist)
				cdict['w'].append(float(datalist[0]))
				cdict['f'].append(float(datalist[1]))
				if clength<3:
					cdict['v'].append(None)
				elif np.isnan(float(datalist[2])):
					cdict['v'].append(None)
				else:
					cdict['v'].append(np.sqrt(np.abs(float(datalist[2]))))
			cfile.close()
			cdict['v'] = self.fill_error(cdict['f'],cdict['v'])
			cdict['f'],cdict['v'] = self.rescale(cdict['f'],cdict['v']\
				,bsnipreds[name])
			if np.min(cdict['w'])>self.pars.wvs[0] or np.max(cdict['w'])\
				<self.pars.re_end:
				del self.data[name]
				continue
			cdict['w'],cdict['f'],cdict['v'] = self.rebin(cdict['w'],cdict['f']\
				,cdict['v'])
			if min(cdict['f'])<=0:
				del self.data[name]
				continue
			self.data[name].append(Spectrum(name,'Bsnip',0,cdict['w']\
				,cdict['f'],cdict['v'],bsnipreds[name]))
			self.keys[name] = None

	def pop_factory(self):
		"""
		Uses Sam's IDRTools to go through the Factory dataset
		"""
		warnings.filterwarnings('error')
		factoryDataSet = IDRTools.Dataset()
		for sne in factoryDataSet.sne:
			self.data[sne.target_name] = []
			for spec in sne.spectra:
				w,f,v = spec.rf_spec()
				if w[0]>self.pars.re_start or w[-1]<self.pars.re_end:
					continue
				w,f,v = self.rebin(w,f,v)
				try:
					new_spec = Spectrum(sne.target_name,'Factory'\
						,spec.salt2_phase,np.array(w),np.array(f),np.sqrt(v)\
						,spec.sn_data['host.zhelio'])
					self.data[sne.target_name].append(new_spec)
				except:
					pass
			if self.data[sne.target_name]!=[]:
				self.keys[sne.target_name] = None
			else:
				del self.data[sne.target_name]

	def pop_phases(self,source_type='Public'):
		"""
		Puts all the novas of source_type into the phase_novas dictionary

		:source_type: (str) from Parameters.sources
		"""
		start_set = {sn:self.data[sn] for sn in self.data \
			if self.data[sn][0].source==source_type}
		for nova in start_set:
			for spec in start_set[nova]:
				if (spec.phase>=(self.pars.phases[0]-self.pars.day_cut)) \
					and (spec.phase<(self.pars.phases[-1]+self.pars.day_cut)):
					self.phase_novas[tools.closest_match(spec.phase\
						,self.pars.phases)].append(spec)

	def pop_public(self):
		"""
		Goes through all the spectra downloaded via the scraper.py script.
		"""
		meta_file = open(self.pars.pubmeta_file,'rb')
		pubmeta = pickle.load(meta_file)
		meta_file.close()
		warnings.filterwarnings('error')
		for nova in pubmeta:
			sn = pubmeta[nova]
			red = np.mean(sn['redshift'])
			broke = False
			for name in sn['names']:
				if name in self.keys:
					broke = True
					break
			if broke:
				break
			self.data[nova] = []
			spectra_file = open(sn['spectra_file'],'rb')
			spectra = pickle.load(spectra_file)
			spectra_file.close()
			for num in spectra:
				spec = spectra[num]
				if len(spec['wvs'])<300:
					continue
				if spec['wvs'][0]>self.pars.wvs[0] or spec['wvs'][-1]\
					<self.pars.wvs[-1]:
					continue
				wvs,flux,errs = spec['wvs'],spec['flux'],spec['errs']
				dust_flux = None
				ebv = None
				if ('deredshifted' not in spec) and ('reduction' not in spec):
					wvs = np.divide(wvs,1+red)
					if wvs[-1]<self.pars.wvs[-1]:
						continue
				for i in range(len(errs)):
					if (errs[i] is not None) and (errs[i]>=flux[i] \
						or errs[i]<=flux[i]/10000):
						errs[i] = None
				if (None in errs) or (np.nan in errs):
					errs = self.fill_error(flux,errs)
					if np.mean(errs)>=(self.pars.max_error*np.mean(flux)):
						errs = [flux[i]*self.pars.max_error for i \
						in range(len(flux))]
				flux,errs = self.rescale(flux,errs,red)
				wvs,flux,errs = self.rebin(wvs,flux,errs)
				if 'ebv' in sn:
					ebv = np.mean(sn['ebv'])
					dust_flux = self.deredden(wvs,flux,np.mean(sn\
						['ebv']))
				if np.min(flux)>0:
					try:
						new_spec = Spectrum(nova,'Public',spec['phase'],wvs\
							,flux,errs,red,dust_flux,ebv)
						self.data[nova].append(new_spec)
					except:
						pass
			if self.data[nova]==[]:
				del self.data[nova]
			else:
				for name in sn['names']:
					self.keys[name] = None

	def rebin(self,wvs,flux,errs=None):
		"""
		Rebins the data to the Parameters.wvs bins

		:wvs: (float array) Angstrom wavelength values
		:flux: (float array) flux values
		:errs: (float array) (optional) error values, with no None elements

		:returns: (wvs,flux,errs) or (wvs,flux) tuple depending on if errs is
				  None or not
		"""
		resample_function = interp1d(wvs,flux,kind='linear')
		flux = resample_function(self.pars.wvs)
		if errs is not None:
			err_resample_function = interp1d(wvs,errs,kind='linear')
			errs = err_resample_function(self.pars.wvs)
			return (self.pars.wvs,flux,errs)
		return (self.pars.wvs,flux)

	def rescale(self,flux,errs,redshift):
		"""
		Fixes the public flux to be at a common distance to the Factory data
		as in Sam's IDRTools.py file (fixes them inplace).

		:flux: (float array) flux values
		:errs: (float array) error values, possibly containing None elements
		:redshift: (float) redshift

		:returns: (float tuple) corrected values in (flux,errs) form
		"""
		dl = (1+redshift)*cosmo.comoving_transverse_distance(redshift).value
		dlref = cosmo.luminosity_distance(0.05).value
		flux = np.divide(flux,(1+redshift)/(1+0.05)*(dl/dlref)**2)
		errs = np.divide(errs,(1+redshift)/(1+0.05)*(dl/dlref)**2)
		return (flux,errs)



class Spectrum(object):
	""" Creates a supernova Spectrum object """
	def __init__(self,key,source,phase,wvs,flux,errs,red,dust_flux=None\
		,ebv=None):
		"""
		Constructor

		:key: (str) the name of the supernova (note aliases are put in 
				SnSet.keys)
		:source: (str) from Parameters.sources
		:phase: (float) phase relative to max light
		:wvs: (float array) wavelengths
		:flux: (float array) flux values
		:errs: (float array) error values
		:red: (float) the redshift
		:dust_flux: (float array) (optional) dust corrected flux values
		:ebv: (float) (optional) the ebv value (for reference)
		"""
		self.pars = Parameters()

		self.key = key
		self.source = source
		self.phase = phase
		self.wvs = wvs
		self.signal = {'flux':flux,'dust_flux':flux}
		self.errs = errs
		if np.mean(self.errs) is np.nan or str(np.mean(self.errs))=='nan':
			self.errs = [self.signal['flux'][i]*self.pars.max_error for i \
				in range(self.pars.num_points)]
		self.red = red
		if dust_flux is not None:
			self.signal['dust_flux'] = dust_flux
		self.ebv = ebv
		self.r_v = 3.1 # for use with r_v variations

		self.fits = {}
		self.coeffs = {}
		self.chi = None # for use with the fitting
		self.rms = None # for use with the fitting
		self.rephase = phase # parameter for use after phase classification

		self.calc_snr()
		self.calc_bv()
		self.populate_signals()

	def calc_bv(self):
		""" Calculates the B-V color difference """
		b_mag = self.integrate_band('bessellb','dust_flux')
		v_mag = self.integrate_band('bessellv','dust_flux')
		self.bv = np.log10(b_mag)-np.log10(v_mag)

	def calc_snr(self):
		""" Calculates the signal to noise ratio of the spectra """
		b,a = scipy.signal.butter(self.pars.butter_order,self.pars.butter_cut)
		smoothed = scipy.signal.filtfilt(b,a,self.signal['flux'])
		noise = tools.fit_rms(smoothed,self.signal['flux'])
		signal = tools.rms(self.signal['flux'])
		self.snr = signal/noise

	def integrate_band(self,band_name,flux_source):
		"""
		Calculates the filtered integrated magnitude of the flux through a band

		:band_name: (str) the name of the band to use with sncosmo.get_bandpass
		:flux_source: (str) the source of the flux--a key of self.signal

		:returns: (float) the integrated magnitude of flux_source through 
					band_name
		"""
		band = sncosmo.get_bandpass(band_name)

		BandFunction = interp1d(band.wave,band.trans,kind='linear')
		new_bins = np.arange(band.wave[0],band.wave[-1]+self.pars.re_step\
			,self.pars.re_step).astype(float)
		new_band = BandFunction(new_bins)
		ws = list(self.wvs)
		bs = list(new_bins)
		if bs[0]>ws[0]:
			bi = 0
			wi = ws.index(bs[0])
		else:
			wi = 0
			bi = bs.index(ws[0])
		if bs[-1]>ws[-1]:
			be = bs.index(ws[-1])
			we = ws.index(ws[-1])
		else:
			be = bs.index(bs[-1])
			we = ws.index(bs[-1])
		return np.sum(np.multiply(self.signal[flux_source][wi:we+1]\
			,new_band[bi:be+1]))*self.pars.re_step

	def normalize(self,mode='b'):
		"""
		Normalizes the Spectrum for comparison purposes to other spectra, where
		the possible modes are 'total' which normalizes the total integral of 
		the flux; 'b' which normalizes across the integral over the 'bessellb'
		band; and 'max' which normalizes the maximum flux of the Spectrum.

		:mode: (str) 'total', 'b', or 'max' as explained above
		"""
		scale = getattr(self.pars,mode+'_flux')
		if mode=='total':
			self.errs = np.multiply(scale/np.sum(self.signal['flux']),self.errs)
			for s in ['','dust_']:
				self.signal[s + 'flux'] = np.multiply(scale/np.sum(self.signal\
					[s + 'flux']),self.signal[s + 'flux'])
				self.signal['filter_' + s + 'flux'] = np.multiply(scale/np.sum\
					(self.signal['filter_' + s + 'flux']),self.signal\
					['filter_' + s + 'flux'])
		elif mode=='b':
			self.errs = np.multiply(scale/self.integrate_band('bessellb'\
				,'flux'),self.errs)
			for s in ['','dust_']:
				self.signal[s + 'flux'] = np.multiply(scale/self.integrate_band\
					('bessellb',s + 'flux'),self.signal[s + 'flux'])
				self.signal['filter_' + s + 'flux'] = np.multiply(scale/self\
					.integrate_band('bessellb','filter_' + s + 'flux')\
					,self.signal['filter_' + s + 'flux'])
		else:
			self.errs = np.multiply(scale/np.max(self.signal['flux']),self.errs)
			for s in ['','dust_']:
				self.signal[s + 'flux'] = np.multiply(scale/np.max(self.signal\
					[s + 'flux']),self.signal[s + 'flux'])
				self.signal['filter_' + s + 'flux'] = np.multiply(scale/np.max\
					(self.signal['filter_' + s + 'flux']),self.signal\
					['filter_' + s + 'flux'])
		if np.mean(self.errs) is np.nan or str(np.mean(self.errs))=='nan':
			self.errs = [self.signal['flux'][i]*self.pars.max_error for i \
				in range(self.pars.num_points)]

	def populate_signals(self):
		""" Populates self.signals with various modes of analysis """
		b,a = scipy.signal.butter(self.pars.butter_order,self.pars.butter_cut)
		for s in ['','dust_']:
			smoothed = scipy.signal.filtfilt(b,a,self.signal[s + 'flux'])
			self.signal['filter_' + s + 'flux'] = smoothed
			self.signal[s + 'mag'] = np.multiply(-2.5,np.log10(self.signal\
				[s + 'flux']))
			self.signal['filter_' + s + 'mag'] = scipy.signal.filtfilt(b,a\
				,self.signal[s + 'mag'])