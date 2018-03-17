"""
Project: pubsne
Authors:
	kceades
	sam-dixon
"""

# imports from standard packages
import os
import requests
import json
import numpy as np
import pandas as pd
import datetime as dt
from astropy.time import Time
from astropy import units as u
import pickle
from tqdm import tqdm
import warnings



class Parameters(object):
	""" Parameters for use in the scraper """
	def __init__(self):
		""" Constructor """
		self.max_red = 0.1
		self.prefix = \
			'https://sne.space/astrocats/astrocats/supernovae/output/json/'
		self.base_dir = os.getcwd()
		self.spec_dir = self.base_dir + '/spectra'
		if not os.path.isdir(self.spec_dir):
			os.makedirs(self.spec_dir)



class Scraper(object):
	""" Automated web scraper to pull data from sne.space and format it """
	def __init__(self):
		""" Constructor """
		self.pars = Parameters()

	def run(self):
		""" Runs the scraper """
		self.load_urls()
		self.pull_data()
	
	def load_urls(self,filename='sne.csv'):
		"""
		Loads the urls based on the csv file saved locally

		:filename: (str) the name of the locally saved csv file with the
					urls for pull data from
		"""
		name_file = os.path.join(self.pars.base_dir,filename)
		names = pd.read_csv(name_file).Name
		self.urls = [self.pars.prefix+name.replace(' ','%20')+'.json' \
			for name in names]

	def pull_data(self,save_name='pubmeta'):
		""" Runs through all the urls and saves the data to a dictionary. """
		self.data = {}
		self.sns = {}
		for url in tqdm(self.urls):
			self.pull_url_data(url)
		if self.data!={} and self.sns!={}:
			save_path = os.path.join(self.pars.base_dir,save_name+'.p')
			save_file = open(save_path,'wb')
			pickle.dump(self.data,save_file)
			save_file.close()

	def pull_url_data(self,url):
		"""
		Runs through the parsing for a single url

		:url: (str) single url to request data from via sne.space
		"""
		r = requests.get(url)
		data = json.loads(r.content)
		sn_key = list(data.keys())[0]
		if sn_key in self.data:
			return
		data = data[sn_key]
		nova_info = self.get_nova_info(data)
		if nova_info is None:
			return
		try:
			spectra = self.get_spectra(data['spectra'],nova_info['max_light']\
				,sn_key)
		except:
			return
		if spectra is None:
			return
		else:
			for name in nova_info['names']:
				if name in self.sns:
					return
			for name in nova_info['names']:
				self.sns[name] = None
			nova_info['phases'] = spectra['phases']
			nova_info['spectra_file'] = spectra['file']
			self.data[sn_key] = nova_info

	def get_nova_info(self,content):
		"""
		Parses the JSON from sne.space for a given supernova

		:content: (dict) the dictionary from sne.space for a given supernova

		:returns: (dict or None) a new dictionary with useful information;
					returns None if certain key pieces of information are
					missing
		"""
		r_dict = {}
		warnings.filterwarnings('error')
		try:
			max_dates = [self.ymd_to_mjd(item['value']) for item \
				in content['maxdate']]
			max_visual_dates = [self.ymd_to_mjd(item['value']) for item \
				in content['maxvisualdate']]
			r_dict['max_time'] = max_dates
			r_dict['max_band'] = [item['value'] for item in content['maxband']]
			r_dict['max_visual_time'] = max_visual_dates
			r_dict['max_visual_band'] = [item['value'] for item \
				in content['maxvisualband']]

			r_dict['max_light'] = max_visual_dates[0]

			r_dict['claimed_type'] = [item['value'] for item \
				in content['claimedtype']]

			reds = [float(item['value']) for item in content['redshift']]
			r_dict['redshift'] = reds
		except:
			return None
		if np.max(r_dict['redshift'])>self.pars.max_red:
			return None
		try:
			dists = []
			for item in content['lumdist']:
				base = float(item['value'])
				if item['u_value']=='Mpc':
					dists.append(base*1e6)
			if dists!=[]:
				r_dict['lumdist'] = dists
		except:
			pass
		try:
			r_dict['sources'] = [x['name'] for x in content['sources']]
		except:
			pass
		try:
			r_dict['names'] = [item['value'] for item in content['alias']]
		except:
			r_dict['names'] = [content['name']]
		try:
			ebvs = [float(item['value']) for item in content['ebv']]
			r_dict['ebv'] = ebvs
		except:
			pass
		try:
			vels = [float(item['value']) for item in content['velocity'] \
				if item['u_value']=='km/s']
			r_dict['velocity'] = vels
		except:
			pass
		try:
			mags = [float(item['value']) for item in content['maxvisualabsmag']]
			r_dict['max_visual_absmag'] = mags
		except:
			pass
		return r_dict

	def get_spectra(self,content,max_time,name):
		"""
		Parses the spectra from sne.space to return a dictionary of nicely
		formatted and useful information

		:content: (list of dicts) list of spectra dictionaries
		:max_time: (float) MJD time for max light
		:name: (str) primary name of the supernova

		:returns: (dict or None) dictionary with the spectra data and useful
					metadata about it;
					returns None if critical bits are missing
		"""
		save_path = os.path.join(self.pars.spec_dir,name + '.p')

		r_dict = {'file':save_path,'phases':[]}
		save_specs = {}

		warnings.filterwarnings('error')
		counter = 0
		for spec in content:
			if ('time' not in spec.keys()) or ('u_fluxes' not in spec.keys()) \
				or (spec['u_wavelengths']!='Angstrom'):
				continue
			spec_dict = {'units':spec['u_fluxes']}
			try:
				spec_dict['phase'] = Time(float(spec['time'])\
					,format=spec['u_time'].lower()).mjd - max_time
			except:
				continue
			try:
				spec_dict['reduction'] = spec['reduction']
			except:
				pass
			try:
				spec_dict['deredshifted'] = spec['deredshifted']
				spec_dict['dereddened'] = spec['dereddened']
			except:
				pass
			min_wv = 0
			max_wv = 1e4
			try:
				min_wv = float(spec['exclude']['below'])
				max_vw = float(spec['exclude']['above'])
			except:
				pass

			spec_dict['wvs'] = []
			spec_dict['flux'] = []
			spec_dict['errs'] = []
			for point in spec['data']:
				wv = float(point[0])
				if wv>=min_wv and wv<=max_wv:
					spec_dict['wvs'].append(wv)
					spec_dict['flux'].append(float(point[1]))
					if len(point)>2:
						spec_dict['errs'].append(float(point[2]))
					else:
						spec_dict['errs'].append(None)

			save_specs[counter] = spec_dict
			counter += 1
			r_dict['phases'].append(spec_dict['phase'])
		if save_specs=={}:
			return None
		else:
			save_file = open(save_path,'wb')
			pickle.dump(save_specs,save_file)
			save_file.close()
			return r_dict

	def ymd_to_mjd(self,date):
		"""
		Converts YYYY/MM/DD date string to MJD

		:date: (str) input string of the form 'YYYY/MM/DD'

		:returns: (float) the MJD date
		"""
		y,m,d = [int(n) for n in date.split('/')]
		return Time(dt.datetime(y,m,d)).mjd



if __name__=='__main__':
	scraper_object = Scraper()
	scraper_object.run()