"""
Project: pubsne
Authors:
	kceades
"""

# importing from files I wrote
import snmc
import tools
import snetools

# imports from standard packages
import os
import numpy as np
from sklearn.decomposition import PCA
from mfa import FactorAnalysis
import pickle
from multiprocessing import Process
from threading import Thread
import time



class Parameters(snmc.Parameters):
	"""
	Storing parameters for the data creation

	note: this extends the snmc.Parameters class, so as a base it has all of
		  its parameters (after the call to it's init)
	"""
	def __init__(self):
		""" constructor """
		# getting all the fields from snmc.Parameters
		snmc.Parameters.__init__(self)
		# the number of components used in the analysis objects
		self.num_components = 15
		# the modes of analysis
		self.model_modes = ['PCA','emFA']
		# the types of signals we can look at
		self.signal_modes = ['flux','filter_flux','dust_flux'\
			,'filter_dust_flux','mag','filter_mag','dust_mag','filter_dust_mag']
		# reduced sources list (to only include factory and public)
		self.main_sources = ['Factory','Public']
		# whether or not to reject outliers based on statistical significance
		# (i.e. x-sigma outliers) when creating an analysis object
		self.reject_outliers = True
		# the number of sigma to do a rejection at
		self.reject_sigma = 4.0
		# the directory where all the pickled data files will be stored
		self.base_dir = os.getcwd() + '/pickleddata'
		if not os.path.isdir(self.base_dir):
			os.makedirs(self.base_dir)



class DataGenerator(object):
	"""
	Class to generate data to be used in analyses: it generates based on the
	signal type you want to give it, the source and the model mode.
	"""
	def __init__(self):
		""" Constuctor """
		self.pars = Parameters()
		self.snset = None
		self.snset_generator = None
		self.a_dict = None
		self.coeff_dict = None

	def run_everything(self,print_progress=True):
		"""
		Goes through all possible source types and creates the various pickled
		dictionaries from them to store for future use.

		:print_progress: (bool) (optional) whether to print progress through the
							for loop
		"""
		for source in self.pars.sources:
			if print_progress:
				print(source)
			self.run_everything_individual(source)

	def run_everything_individual(self,source):
		"""
		Creates the pickled dictionaries with supernovae and analysis objects
		for use later.

		:source: (str) choose from snmc.Parameters.sources
		"""
		self.create_novas(source)
		self.create_phase_novas(source)
		self.run_full_analysis(source,'dust_flux')
		# self.run_full_analysis(source,'filter_dust_flux')

	def create_phase_novas(self,source_type='Public'):
		"""
		creates a pickled file of supernovas by phase
		
		:source_type: (str) choose from snmc.Parameters.sources
		"""
		if self.snset_generator!=source_type:
			self.snset = snmc.SnSet(source_type)
			self.snset_generator = source_type
		filename = os.path.join(self.pars.base_dir,source_type+'_Novas_Phase.p')
		savefile = open(filename,'wb')
		pickle.dump(self.snset.phase_novas,savefile)
		savefile.close()

	def create_novas(self,source_type='Public'):
		"""
		creates a pickled nova file with all the supernova objects

		:source_type: (str) choose from snmc.Parameters.sources
		"""
		if self.snset_generator!=source_type:
			self.snset = snmc.SnSet(source_type)
			self.snset_generator = source_type
		filename = os.path.join(self.pars.base_dir,source_type+'_Novas.p')
		savefile = open(filename,'wb')
		pickle.dump(self.snset.data,savefile)
		savefile.close()

	def run_full_analysis(self,source_type='Public',signal_mode='dust_flux'):
		"""
		creates a pickled dictionary of analysis objects

		:source_type: (str) choose from snmc.Parameters.sources
		:signal_mode: (str) choose from Parameters.signal_modes
		"""
		if self.snset_generator!=source_type:
			self.create_phase_novas(source_type)

		self.a_dict = {mode:{phase:None for phase \
			in self.pars.phases} for mode in self.pars.model_modes}
		self.coeff_dict = {mode:{phase:None for phase \
			in self.pars.phases} for mode in self.pars.model_modes}

		for mode in self.pars.model_modes:
			for phase in self.pars.phases:
				self.single_run(mode,self.pars.num_components,phase,signal_mode)
		
		filename = ''
		if signal_mode[:6]=='filter':
			filename = os.path.join(self.pars.base_dir,source_type\
				+'_Objects_Filter.p')
		else:
			filename = os.path.join(self.pars.base_dir,source_type+'_Objects.p')
		savefile = open(filename,'wb')
		pickle.dump(self.a_dict,savefile)
		savefile.close()

		coeffname = ''
		if signal_mode[:6]=='filter':
			coeffname = os.path.join(self.pars.base_dir,source_type\
				+'_Coeffs_Filter.p')
		else:
			coeffname = os.path.join(self.pars.base_dir,source_type+'_Coeffs.p')
		coefffile = open(coeffname,'wb')
		pickle.dump(self.coeff_dict,coefffile)
		coefffile.close()

	def single_run(self,mode='emFA',num=15,phase=0,signal_mode='dust_flux'):
		"""
		Creates the analysis object to populate self.a_dict.

		:mode: (str) choose from Parameters.model_modes
		:num: (int) usually called with Parameters.num_components as the input
		:phase: (int) choose from snmc.Parameters.phases
		:signal_mode: (str) choose from Parameters.signal_modes
		"""
		spectra = self.snset.phase_novas[phase]
		sigma = self.pars.reject_sigma
		if self.pars.reject_outliers is False:
			sigma = 1000
		a_object,coeffs = snetools.create_model(spectra,signal_mode,mode\
			,num,sigma)
		self.a_dict[mode][phase] = a_object
		self.coeff_dict[mode][phase] = coeffs



def source_create_everything(source):
	"""
	Creates a DataGenerator object for the specified source and runs through
	all the data and analysis object creation for that object.

	:source: (str) choose from Parameters.sources
	"""
	d = DataGenerator()
	d.run_everything_individual(source)



def processed_run(print_progress=True):
	"""
	Speeds up creating the data for all the sources by doing them individually
	in different processes.

	:print_progress: (bool) (optional) whether or not to print the source while
						running through it
	"""
	pars = Parameters()
	processes = []
	for source in pars.sources:
		process = Process(target=source_create_everything,args=(source,))
		processes.append(process)
		process.start()
		if print_progress:
			print('Process started for {} source'.format(source))
	for process in processes:
		process.join()
	if print_progress:
		print('All processes finished.')



def threaded_run(print_progress=True):
	"""
	Speeds up creating the data for all the sources by doing them individually
	in different threads.

	:print_progress: (bool) (optional) whether or not to print the source while
						running through it
	"""
	pars = Parameters()
	threads = []
	for source in pars.sources:
		thread = Thread(target=source_create_everything,args=(source,))
		threads.append(thread)
		thread.start()
		if print_progress:
			print('Thread started for {} source'.format(source))
	for thread in threads:
		thread.join()
	if print_progress:
		print('All threads finished.')



if __name__ == '__main__':
	# based on my trials, it looks like processed_run runs significantly faster
	# than threaded_run, and both are faster than creating a DataGenerator
	# object and calling the run_everything method.
	processed_run()