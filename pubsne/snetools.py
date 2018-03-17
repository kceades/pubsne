"""
Project: pubsne
Authors:
	kceades
"""

# imports from files I wrote
import snmc
import tools
import mlalgs

# import from standard packages
import numpy as np
from sklearn.decomposition import PCA
from mfa import FactorAnalysis
import extinction
import matplotlib.pyplot as plt

def create_coeff_dict(spectra,signal_mode,a_object):
	"""
	Creates a dictionary with keys 'avg' and 'std' that could be used with
	simulated spectra for example. These keys will have corresponding values
	that are the average and standard deviation of coefficients from the
	training spectra on the analysis object (an emFA or PCA).

	:spectra: (list of snmc.Spectrum objects) the spectra that were used in
				training the analysis object
	:signal_mode: (str) the type of signal mode to use
	:a_object: (FactorAnalysis or PCA object) the object that was trained

	:returns: (dict) coefficient dictionary as described above
	"""
	CM = a_object.transform([spec.signal[signal_mode] for spec in spectra])
	return {'avg':np.mean(CM,axis=0),'std':np.std(CM,axis=0)}

def create_model(spectra,signal_mode,a_type,num_components,reject_sigma=5):
	"""
	Creates an analysis object based on the source spectra with the specified
	signal and number of components. Iterates until there are no outliers (rms)
	above the sigma rejection level or until there are fewer than 5 more sne
	than the number of components.

	:spectra: (list of snmc.Spectrum objects) sne spectra to be used
	:signal_mode: (str) the type of signal mode to use
	:a_type: (str 'emFA' or 'PCA') the type of analysis object to use
	:num_components: (int) the number of components to use in the model
	:reject_sigma: (float) the sigma outlier rejection level

	:returns: (tuple of (a_type object,coeff_dict)) returns a tuple where the
				first element is the analysis object (model, but could be None)
				and the second is a coefficient dictionary with keys of 'avg'
				and 'std'
	"""
	while True:
		sne_count = len(spectra)
		if sne_count<5+num_components:
			return (None,None)
		a_object = None
		if a_type=='PCA':
			a_object = PCA(n_components=num_components)
		elif a_type=='emFA':
			a_object = FactorAnalysis(n_components=num_components)
		else:
			raise TypeError(''.join(['{} is not a valid model '.format(a_type)\
				,'. Choose from "emFA" or "PCA".']))

		a_object.fit([spec.signal[signal_mode] for spec in spectra])
		for spec in spectra:
			fit_spectrum(spec,signal_mode,a_object)
		rms_threshold = np.mean([spec.rms for spec in spectra]) \
			+ reject_sigma*np.std([spec.rms for spec in spectra])
		spectra = [spec for spec in spectra if spec.rms<rms_threshold]
		if len(spectra)==sne_count:
			return (a_object,create_coeff_dict(spectra,signal_mode,a_object))

def deredden(spectrum,r_v):
	"""
	Removes the extinction from the supernova and repopulates the signal types.
	Note that this function overrides the data stored in the spectrum object
	and creates a new set of data based on the new r_v value.

	:spectrum: (snmc.Spectrum object) the supernova to deredden
	:r_v: (float) the r_v value to use in the dereddening
	"""
	dust = extinction.ccm89(spectrum.wvs,r_v*spectrum.ebv,r_v)
	exp_dust = np.power(10,np.multiply(-0.4,dust))
	spectrum.signal['dust_flux'] = np.divide(spectrum.signal['flux'],exp_dust)
	spectrum.populate_signals()
	spectrum.normalize()

def find_outliers(spectra,signal_mode,a_object,reject_sigma):
	"""
	Finds outliers from the input spectra based on fitting to the analysis
	object and flagging anything about reject_sigma from the norm (rms). Note
	this function should really only be called for sne at the same phase.

	:spectra: (list of snmc.Spectrum objects) sne spectra to check outliers of
	:signal_mode: (str) the type of signal mode to use
	:a_object: (PCA or FA object) analysis object that should probably have been
				trained using the same signal mode and at the same phase
	:reject_sigma: (float) the sigma outlier rejection level

	:returns: (set of str) set of supernova keys that are outliers and likely
				need to be a combination of rephased or have r_v values checked
	"""
	for spec in spectra:
		fit_spectrum(spec,signal_mode,a_object)
	rms_threshold = np.mean([spec.rms for spec in spectra]) \
		+ reject_sigma*np.std([spec.rms for spec in spectra])
	return set([spec.key for spec in spectra if spec.rms>=rms_threshold])

def fit_phase(spectrum,signal_mode,phaser):
	"""
	Fits the phase for the input spectrum (which is presumably an outlier in
	the phased PCA fits) under the given signal type to the set of 'good'
	(properly phased) spectra. It stores this new phase under the spectrum's
	self.rephase property.

	:spectrum: (snmc.Spectrum object) the spectrum to fit the phase of
	:signal_mode: (str) the type of signal mode to use
	:phaser: (phaser object) the phaser to use for fitting
	"""
	fit = phaser.fit(spectrum.signal[signal_mode])
	spectrum.rephase = fit.value

def fit_rv(spectrum,signal_mode,a_object):
	"""
	Attempts to vary the r_v value for the input spectrum to see if a better
	fit to the model can be achieved. Stores this in the spectrum's self.r_v.
	*for now, assumes the normalization is to have average flux of 1*

	:spectrum: (snmc.Spectrum object) the spectrum to vary the r_v of
	:signal_mode: (str) the type of signal mode to use
	:a_object: (PCA or FA object) analysis object that should probably have been
				trained using the same signal mode and at the same phase
	"""
	best_rms = 1000
	best_r_v = 3.1
	r_v_array = np.arange(0.1,10.1,0.1)
	for r_v in r_v_array:
		deredden(spectrum,r_v)
		fit_spectrum(spectrum,signal_mode,a_object)
		if spectrum.rms<best_rms:
			best_rms = spectrum.rms
			best_r_v = r_v
	spectrum.r_v = best_r_v
	deredden(spectrum,spectrum.r_v)
	fit_spectrum(spectrum,signal_mode,a_object)

def fit_spectrum(spectrum,signal_mode,a_object):
	"""
	Fits the input spectrum with the specific signal to the analysis object.

	:spectrum: (snmc.Spectrum object) sne spectrum to be fit
	:signal_mode: (str) the type of signal mode to use
	:a_object: (PCA or FA object) analysis object that should probably have been 
				trained using the same signal mode (unless comparisons are
				being made)
	"""
	PCs = a_object.components_
	CM = a_object.transform([spectrum.signal[signal_mode]])
	bestfit = a_object.mean_
	spectrum.fits['mean'] = bestfit
	for j in range(len(PCs)):
		bestfit = np.add(bestfit,np.multiply(PCs[j],CM[0][j]))
		spectrum.fits[j] = (CM[0][j],bestfit)
	spectrum.rms = tools.fit_rms(bestfit,spectrum.signal[signal_mode])
	spectrum.chi = tools.fit_chi(bestfit,spectrum.signal[signal_mode]\
		,spectrum.errs,len(bestfit)-len(PCs))