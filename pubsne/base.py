"""
Project: pubsne
Authors:
	kceades
"""

# imports from classes I wrote
import snmc
import creator
import tools
import snetools

# imports from standard packages
import os
import numpy as np
from sklearn.decomposition import PCA
from mfa import FactorAnalysis
import pickle
import matplotlib.pyplot as plt
import scipy.signal
import sncosmo
from multiprocessing import Process



class Abase(object):
	"""
	Abase (analysis base) object that gets the data created by the creator.py
	script and stored locally. It then has multiple methods to do various types
	of analysis with that data. One of the highlights is that it can create a
	series of html files and images to view a full gallery of fitting across
	all the supernova using different training and target sources.
	"""
	def __init__(self):
		""" Constructor for the Abase (analysis base) object """
		self.pars = creator.Parameters()
		self.novas = {}
		self.phase_novas = {}
		self.a_objects = {}
		self.a_coeffs = {}

		self.base_dir = os.getcwd() + '/results'
		if not os.path.isdir(self.base_dir):
			os.makedirs(self.base_dir)

		self.pop_novas()
		self.pop_objects()
		self.pop_coeffs()

		self.pop_phases()

		self.exp_dict = {}

	def pop_novas(self):
		""" Populates the novas dictionary """
		for source in self.pars.sources:
			novaname = os.path.join(self.pars.base_dir,source+'_Novas.p')
			novafile = open(novaname,'rb')
			self.novas[source] = pickle.load(novafile)
			novafile.close()

	def pop_objects(self):
		""" Populates the object dictionary """
		for source in self.pars.sources:
			object_name = os.path.join(self.pars.base_dir,source+'_Objects.p')
			object_file = open(object_name,'rb')
			self.a_objects[source] = pickle.load(object_file)
			object_file.close()

	def pop_coeffs(self):
		""" Populates the coefficients dictionary """
		for source in self.pars.sources:
			coeff_name = os.path.join(self.pars.base_dir,source+'_Coeffs.p')
			coeff_file = open(coeff_name,'rb')
			self.a_coeffs[source] = pickle.load(coeff_file)
			coeff_file.close()

	def pop_phases(self):
		""" creates a dictionary of novas by phase """
		for source in self.pars.sources:
			phasename = os.path.join(self.pars.base_dir,source+'_Novas_Phase.p')
			phasefile = open(phasename,'rb')
			self.phase_novas[source] = pickle.load(phasefile)
			phasefile.close()

	def web_components(self,analysis_mode,source):
		"""
		creates the plots for and saves the html page of the components of a
		model

		:analysis_mode: (str) choose from creator.Parameters.model_modes
		:source: (str) choose from snmc.Parameters.sources
		"""
		comps = self.a_objects[source][analysis_mode][0].components_
		webstring = """<!DOCTYPE html>

<html lang='en-US'>



<head>


<title>Principle Components: """ + ' '.join([source,analysis_mode])\
	+ """ Training</title>
<meta charset='UTF-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<link rel='stylesheet' href='sitestyle.css'>


</head>



<body>

<h1>""" + ' '.join([source,analysis_mode,'Components']) + """</h1>

"""
		plt.ioff()
		savedir = ''.join([self.base_dir,'/images','_',analysis_mode])
		if not os.path.isdir(savedir):
			os.makedirs(savedir)
		source_dir = ''.join([savedir,'/',source])
		if not os.path.isdir(source_dir):
			os.makedirs(source_dir)
		comps_dir = ''.join([source_dir,'/comps'])
		if not os.path.isdir(comps_dir):
			os.makedirs(comps_dir)
		for j in range(self.pars.num_components):
			savepath = os.path.join(comps_dir,''.join(['comp_{}'.format(j)\
				,'.png']))

			plt.figure()
			plt.plot(self.pars.wvs,comps[j])
			plt.xlabel('Wavelength (Angstroms)')
			plt.ylabel('Flux (arbitrary units)')
			plt.title('Component {}'.format(j))
			plt.savefig(savepath)
			plt.close()

			webstring = webstring + """
<img src='""" + savepath + """' alt='{} Component""".format(j) + """'>
"""

		webstring = webstring + """

</body>

</html>
"""
		savesite = open(os.path.join(source_dir,'comps.html'),'w')
		savesite.write(webstring)
		savesite.close()

	def gallery_images(self,analysis_mode,source,target):
		"""
		creates dictionaries of the target supernovas being fitted to the source
		generated model

		:analysis_mode: (str) choose from creator.Parameters.model_modes
		:source: (str) choose from snmc.Parameters.sources
		:target: (str) choose from snmc.Parameters.sources
		"""
		target_novas = self.phase_novas[target][0]
		source_object = self.a_objects[source][analysis_mode][0]
		num_novas = len(target_novas)

		coeffs = []
		for i in range(num_novas):
			snetools.fit_spectrum(target_novas[i],'dust_flux',source_object)
			coeffs.append([target_novas[i].fits[j][0] for j in \
				range(self.pars.num_components)])
		chi_sort = [(target_novas[i].chi,target_novas[i],coeffs[i]) for i \
			in range(num_novas)]
		chi_sort.sort(key=lambda x:x[0])
		ls_sort = [(target_novas[i].rms,target_novas[i],coeffs[i]) for i \
			in range(num_novas)]
		ls_sort.sort(key=lambda x:x[0])

		self.worst_chis = chi_sort
		self.worst_ls = ls_sort

	def gallery_maker(self,stat_mode,analysis_mode,source,target):
		"""
		create the full gallery of supernova images, and publish the html

		:stat_mode: (str) 'rms' or 'chi'
		:analysis_mode: (str) choose from creator.Parameters.model_modes
		:source: (str) choose from snmc.Parameters.sources
		:target: (str) choose from snmc.Parameters.sources
		"""
		webstring = """<!DOCTYPE html>

<html lang='en-US'>



<head>


<title>Supernova Fitting: """ + ' '.join([source,'Training Applied to',target])\
+ """</title>
<meta charset='UTF-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<link rel='stylesheet' href='sitestyle.css'>


</head>



<body>

<h1>""" + ' '.join([source,'Training Applied to',target]) + """ Novas</h1>

<h1>""" + ' '.join([stat_mode,analysis_mode]) + """</h1>

"""
		plt.ioff()
		self.gallery_images(analysis_mode,source,target)
		savedir = ''.join([self.base_dir,'/images','_',analysis_mode])
		if not os.path.isdir(savedir):
			os.makedirs(savedir)
		source_dir = ''.join([savedir,'/',source])
		if not os.path.isdir(source_dir):
			os.makedirs(source_dir)
		target_dir = ''.join([source_dir,'/',target])
		if not os.path.isdir(target_dir):
			os.makedirs(target_dir)

		if stat_mode=='chi':
			looking_at = self.worst_chis
		else:
			looking_at = self.worst_ls
		distribution = [x[0] for x in looking_at]
		plt.hist(distribution,25,facecolor='green')
		histsavepath = os.path.join(target_dir,''.join(['histogram_',stat_mode\
			,'.png']))
		plt.xlabel(stat_mode)
		plt.ylabel('Number of Supernova')
		plt.title('Fitted Supernova ' + stat_mode + ' Distribution')
		plt.savefig(histsavepath)
		plt.close()
		webstring = webstring + """
<img src='""" + histsavepath + """' alt='Distribution'>
"""
		for x in looking_at:
			res = np.subtract(x[1].signal['dust_flux'],x[1].fits\
				[self.pars.num_components-1][1])

			fsavepath = os.path.join(target_dir,''.join([x[1].key,'_Fit.png']))
			ressavepath = os.path.join(target_dir,''.join([x[1].key\
				,'_Res.png']))
			coeffsavepath = os.path.join(target_dir,''.join([x[1].key\
				,'_Coeffs.png']))

			plt.figure()
			plt.plot(self.pars.wvs,x[1].signal['dust_flux'],color='black'\
				,label='data')
			plt.plot(self.pars.wvs,x[1].fits[self.pars.num_components-1][1]\
				,color='green',label='fit')
			plt.xlabel('Wavelength (Angstroms)')
			plt.ylabel('Flux (Arbitrary Units)')
			plt.title(x[1].key + ' Fitting')
			plt.legend()
			plt.savefig(fsavepath)
			plt.close()

			plt.figure()
			plt.plot(self.pars.wvs,res)
			plt.xlabel('Wavelength (Angstroms)')
			plt.ylabel('Flux (Arbitrary Units)')
			plt.title(x[1].key + ' Residual After Fitting')
			plt.savefig(ressavepath)
			plt.close()

			plt.figure()
			plt.plot(np.arange(self.pars.num_components),x[2],'o')
			plt.xlabel('Component Number')
			plt.ylabel('Coefficient')
			plt.title('Coefficients for ' + x[1].key)
			plt.savefig(coeffsavepath)
			plt.close()

			webstring = webstring + """
<div class='gallery'>
<p>""" + x[1].key + """
<br />
""" + ' '.join([stat_mode,'of',str(x[0])[:6]]) + """</p>
<img src='""" + fsavepath + """' alt='Supernova fit'>
<img src='""" + ressavepath + """' alt='Residual for supernova fit'>
<img src='""" + coeffsavepath + """' alt='Coefficients of the fit'>
</div>
"""
		webstring = webstring + """

</body>



</html>"""
		webname = os.path.join(''.join([self.base_dir,'/images_'\
			,analysis_mode]),''.join([source,'_',target,'_',stat_mode\
			,'_site.html']))
		webfile = open(webname,'w')
		webfile.write(webstring)
		webfile.close()

	def white_noise(self,source='Public',target='Factory',phase=0\
		,model_mode='PCA'):
		"""
		Tests to see whether the residual after fitting target supernova with
		source training components of model_mode type at phase phase is white
		noise or not. Saves the plots to the base_dir folder in a subfolder
		called white_noise.

		:source: (str) choose from snmc.Parameters.sources
		:target: (str) choose from snmc.Parameters.sources
		:phase: (int) choose from snmc.Parameters.phases
		:model_mode: (str) choose from creator.Parameters.model_modes
		"""
		wn_dir = self.base_dir + '/white_noise'
		if not os.path.isdir(wn_dir):
			os.makedirs(wn_dir)
		phase_dir = wn_dir + '/{}'.format(phase)
		if not os.path.isdir(phase_dir):
			os.makedirs(phase_dir)
		# first reduce the target data to the residual after fitting
		spectra = self.phase_novas[target][phase]
		a_object = self.a_objects[source][model_mode][phase]
		data = []
		for spec in spectra:
			snetools.fit_spectrum(spec,'dust_flux',a_object)
			data.append(np.subtract(spec.signal['dust_flux']\
				,spec.fits[self.pars.num_components-1][1]))
		plt.ioff()
		numcorrs = 16
		numdata = len(data)

		autocorrs = {num:[] for num in range(numcorrs)}
		rcorrs = {num:[] for num in range(numcorrs)}
		r = [np.random.randn(self.pars.num_points) for x in range(numdata)]
		for x in data:
			for num in range(numcorrs):
				d = [x[i]*x[i+num] for i in range(self.pars.num_points-num)]
				ad = np.abs(d)
				autocorrs[num].append(np.mean(d)/np.mean(ad))
		for x in r:
			for num in range(numcorrs):
				dr = [x[i]*x[i+num] for i in range(self.pars.num_points-num)]
				adr = np.abs(dr)
				rcorrs[num].append(np.mean(dr)/np.mean(adr))
		dstats = {'avg':[np.mean(autocorrs[num]) for num in autocorrs]\
			,'std':[np.std(autocorrs[num]) for num in autocorrs]}
		rstats = {'avg':[np.mean(rcorrs[num]) for num in rcorrs]\
			,'std':[np.std(rcorrs[num]) for num in rcorrs]}

		pd = {'data':{'mid':dstats['avg'],'high':np.add(dstats['avg']\
			,dstats['std']),'low':np.subtract(dstats['avg'],dstats['std'])}\
			,'rand':{'mid':rstats['avg'],'high':np.add(rstats['avg']\
			,rstats['std']),'low':np.subtract(rstats['avg'],rstats['std'])}}

		fig,ax = plt.subplots()
		colors = ['green','blue']
		baseline = np.arange(numcorrs)
		for i,m in enumerate(pd):
			ax.fill_between(baseline,pd[m]['mid'],pd[m]['high'],color=colors[i]\
				,alpha=0.3)
			ax.fill_between(baseline,pd[m]['mid'],pd[m]['low'],color=colors[i]\
				,alpha=0.3)
			ax.plot(baseline,pd[m]['mid'],'o-',color=colors[i],label=m)
		ax.set_xlabel('Look Forward Num')
		ax.set_ylabel('Correlation')
		ax.set_title('Autocorrelation Comparison: Data vs Random')
		savepath = os.path.join(phase_dir,'whitenoisetest_{}_{}.png'\
			.format(source,target))
		plt.legend()
		plt.savefig(savepath)
		plt.close()

	def make_white_noise_plots(self):
		for source in self.pars.main_sources:
			for target in self.pars.main_sources:
				for phase in self.pars.phases:
					for model_mode in self.pars.model_modes:
						self.white_noise(source,target,phase,model_mode)

	def pub_comp_breakdown(self):
		"""
		Creates a plot of trying to fit the factory components to the public
		ones and looking at the residual compared to the original signal. In
		other words, this looks at how much of each factory component is
		captured by the public components. Saves the results in the
		self.base_dir folder in a subfolder called fac_breakdown.
		"""
		plt.ioff()
		savedir = self.base_dir + '/fac_breakdown'
		if not os.path.isdir(savedir):
			os.makedirs(savedir)
		for phase in self.pars.phases:
			phase_dir = savedir + '/' + str(phase)
			if not os.path.isdir(phase_dir):
				os.makedirs(phase_dir)
			for mode in self.pars.model_modes:
				mode_dir = phase_dir + '/' + mode
				if not os.path.isdir(mode_dir):
					os.makedirs(mode_dir)
				pub_obj = self.a_objects['Public'][mode][phase]
				fac_obj = self.a_objects['Factory'][mode][phase]
				if (fac_obj is not None) and (pub_obj is not None):
					pubPCs = pub_obj.components_
					facPCs = fac_obj.components_
					fitdata = []
					for i,x in enumerate(facPCs):
						fit  = pub_obj.mean_
						CM = pub_obj.transform([x])
						for j in range(15):
							fit = fit + np.multiply(CM[0][j],pubPCs[j])
						residual = np.subtract(x,fit)
						plt.figure()
						plt.xlabel('Wavelength (Angstroms)')
						plt.ylabel('Flux (arbitrary units)')
						plt.title('Original Factory Comp #' + str(i) + \
							' and Res After Public Fit')
						plt.plot(self.pars.wvs,x,color='blue',linewidth=1.0\
							,label='Public Component')
						plt.plot(self.pars.wvs,residual,color='green'\
							,linewidth=1.0,label='Residual')
						plt.legend()
						savepath = os.path.join(mode_dir,'residual_' + str(i) \
							+ '.png')
						plt.savefig(savepath)
						plt.close()
						absres = np.mean(np.abs(residual))
						abssig = np.mean(np.abs(x))
						fitdata.append(absres/abssig)
					plt.figure()
					plt.xlabel('Comp Number')
					plt.ylabel('Residual/Signal')
					plt.title('Residual Divided by Signal of Factory ' + \
						'Comps Fitted to Public')
					plt.plot(np.arange(i+1),fitdata,'o',color='green')
					savepath = os.path.join(mode_dir,'fittedcomps.png')
					plt.savefig(savepath)
					plt.close()

	def explained_variance(self,analysis_mode,source,target):
		"""
		Creates a plot of the explained variance by number of component vectors
		used in the model

		:analysis_mode: (str) choose from creator.Parameters.model_modes
		:source: (str) choose from snmc.Parameters.sources
		:target: (str) choose from snmc.Parameters.sources
		"""
		savedir = ''.join([self.base_dir,'/variance'])
		if not os.path.isdir(savedir):
			os.makedirs(savedir)
		modedir = ''.join([savedir,'/',analysis_mode])
		if not os.path.isdir(modedir):
			os.makedirs(modedir)
		var_dict = {}
		exp_var_dict = {}

		if analysis_mode not in self.exp_dict:
			self.exp_dict[analysis_mode] = {}

		for phase in self.pars.phases:
			if phase not in self.exp_dict[analysis_mode]:
				self.exp_dict[analysis_mode][phase] = {}
			if source not in self.exp_dict[analysis_mode][phase]:
				self.exp_dict[analysis_mode][phase][source] = {}
			if target not in self.exp_dict[analysis_mode][phase][source]:
				self.exp_dict[analysis_mode][phase][source][target] = None

			source_object = self.a_objects[source][analysis_mode][phase]
			if source_object is None:
				continue
			novas = self.phase_novas[target][phase]
			for i in range(len(novas)):
				snetools.fit_spectrum(novas[i],'dust_flux',source_object)

			phasedir = ''.join([modedir,'/',str(phase)])
			if not os.path.isdir(phasedir):
				os.makedirs(phasedir)
			by_bin = [[x.signal['dust_flux'][i] for x in novas] \
				for i in range(self.pars.num_points)]
			var_dict['initial'] = np.mean([np.var(x) for x in by_bin])
			for j in range(self.pars.num_components):
				by_bin = [[x.signal['dust_flux'][i]-x.fits[j][1][i] for x \
					in novas] for i in range(self.pars.num_points)]
				var_dict[j] = np.mean([np.var(x) for x in by_bin])
				exp_var_dict[j] = 1.-var_dict[j]/var_dict['initial']
			self.exp_dict[analysis_mode][phase][source][target] \
				= [exp_var_dict[j] for j in range(self.pars.num_components)]
			plt.ioff()
			plt.figure()
			plt.plot(np.arange(1,self.pars.num_components+1),[exp_var_dict[j] \
				for j in range(self.pars.num_components)],'o-')
			plt.ylim(0,1)
			plt.grid(b=True,axis='y',which='major')
			plt.xlabel('Number of Components Used')
			plt.ylabel('Explained Variance: Peak {:.2f}'.format(exp_var_dict\
				[self.pars.num_components-1]))
			plt.title(' '.join(['Phase {}'.format(phase),'For',source\
				,'Training on',target]))
			savepath = os.path.join(phasedir,''.join([source,'_',target\
				,'_expvar.png']))
			plt.savefig(savepath)
			plt.close()

	def dust_plot(self,key,phase=0,source='Public'):
		"""
		Creates and saves a plot showing how the dust corrections affect the
		signal for an individual supernova.

		:key: (str) name of the supernova
		:phase: (float) (optional) the phase that you want to plot the nearest
				match to of key supernova
		:source: (str) (optional) choose from snmc.Parameters.souces
		"""
		all_specs = self.novas[source][key]
		closest_phase = tools.closest_match(phase,[spec.phase for spec in \
			all_specs])
		spec = all_specs[all_specs.index(closest_phase)]
		plt.ioff()
		plt.figure()
		plt.plot(self.pars.wvs,spec.signal['flux'],color='black'\
			,linewidth=2.0,label='Original')
		plt.plot(self.pars.wvs,spec.signal['dust_flux'],color='green'\
			,linewidth=2.0,label='Corrected')
		plt.xlabel('Wavelength (Angstroms)')
		plt.ylabel('Flux (arbitrary units)')
		plt.title('Dust Correction on Supernova ' + key)
		plt.legend()
		savepath = os.path.join(self.base_dir,key + '_dust.png')
		plt.savefig(savepath)
		plt.close()

	def filter_plot(self,key,phase=0,source='Public'):
		"""
		Creates and saves a plot showing how the current filter affects the
		data for an individual supernova.

		:key: (str) name of the supernova
		:phase: (float) (optional) the phase that you want to plot the nearest
				match to of key supernova
		:source: (str) (optional) choose from snmc.Parameters.sources
		"""
		all_specs = self.novas[source][key]
		closest_phase = tools.closest_match(phase,[spec.phase for spec in \
			all_specs])
		spec = all_specs[all_specs.index(closest_phase)]
		plt.ioff()
		plt.figure()
		plt.plot(self.pars.wvs,spec.signal['dust_flux'],color='black'\
			,linewidth=2.0,label='Original')
		plt.plot(self.pars.wvs,spec.signal['filter_dust_flux'],color='green'\
			,linewidth=2.0,label='Filtered')
		plt.xlabel('Wavelength (Angstroms)')
		plt.ylabel('Flux (arbitrary units)')
		plt.title('Filtering Supernova ' + key)
		plt.legend()
		savepath = os.path.join(self.base_dir,key + '_filter.png')
		plt.savefig(savepath)
		plt.close()


	def run_everything(self,print_progress=True):
		"""
		Creates all the web galleries and also the explained variance plots

		:print_progress: (bool) (optional) whether or not to print which of the
							three big steps it is on
		"""
		if print_progress:
			print('Making component plots')
		self.make_component_plots()
		if print_progress:
			print('Making gallery plots')
		self.make_gallery_plots()
		if print_progress:
			print('Making explained variance plots')
		self.individual_explained_variance_plots()

	def make_component_plots(self,print_progress=True):
		"""
		Makes the plots and overall webpage for the components of each model
		with each source and signal type (filtered or unfiltered)

		:print_progress: (bool) (optional) whether or not to print the progress
							through the nested for loop
		"""
		for source in self.pars.main_sources:
			for model_mode in self.pars.model_modes:
				if print_progress:
					print(source,model_mode)
				self.web_components(model_mode,source)

	def make_gallery_plots(self,print_progress=True):
		"""
		Makes the web galleries for every possible combination of stat_mode,
		model_mode, source,and target

		:print_progress: (bool) (optional) whether or not to print the progress
							through the nested for loop
		"""
		for stat_mode in ['chi','rms']:
			for model_mode in self.pars.model_modes:
				for source in self.pars.main_sources:
					for target in self.pars.main_sources:
						if print_progress:
							print(stat_mode,model_mode,source,target)
						self.gallery_maker(stat_mode,model_mode,source,target)
	
	def individual_explained_variance_plots(self,print_progress=True):
		"""
		Makes the plots for the explained variance curves for every possible
		combination of source, target, and model_mode

		:print_progress: (bool) whether or not to print the progress through
							the nested for loop
		"""
		for source in self.pars.main_sources:
			for target in self.pars.main_sources:
				for model_mode in self.pars.model_modes:
					if print_progress:
						print(source,target,model_mode)
					self.explained_variance(model_mode,source,target)

	def group_explained_variance_plots(self,print_progress=True):
		"""
		Makes and saves the plots for the explained variance curves akin to the
		individual_explained_variance_plots function above, but then also
		makes group plots that have multiple curves on one for close
		comparison.

		:print_progress: (bool) (optional) whether or not to print the progress
							as we work through the groups
		"""
		if print_progress:
			print('doing the individual explained variance curves')
		for model_mode in self.pars.model_modes:
			for i in range(2):
				if print_progress:
					print(self.pars.main_sources[i],self.pars.main_sources\
						[1-i],model_mode)
				self.explained_variance(model_mode,self.pars.main_sources\
					[i],self.pars.main_sources[1-i])

		plt.ioff()
		if print_progress:
			print('doing the group explained variance curves')
		for mode in self.exp_dict:
			for phase in self.pars.phases:
				plt.figure()
				for first in self.exp_dict[mode][phase]:
					for second in self.exp_dict[mode][phase][first]:
						plt.plot(np.arange(1,self.pars.num_components+1)\
							,self.exp_dict[mode][phase][first][second]\
							,label=' '.join(['trained on',first,'applied to'\
								,second]))
				plt.legend()
				plt.xlabel('Number of Components Used')
				plt.ylabel('Explained Variance')
				plt.title(' '.join([mode,'Explained Variance Curves at Phase'\
					,str(phase)]))
				plt.savefig(os.path.join(self.base_dir,'_'.join([mode\
					,str(phase)])))
				plt.close()

	def supernova_phase_distribution(self):
		"""
		Creates and saves a histogram of how many supernova we have per phase
		bin for the 'Factory' and 'Public' supernova.
		"""
		fac_phase_sne = [len(self.phase_novas['Factory'][phase]) for phase \
			in self.pars.phases]
		pub_phase_sne = [len(self.phase_novas['Public'][phase]) for phase \
			in self.pars.phases]
		plt.ioff()
		plt.plot(self.pars.phases,fac_phase_sne,'o-',color='red',linewidth=2.0\
			,label='Factory')
		plt.plot(self.pars.phases,pub_phase_sne,'o-',color='green'\
			,linewidth=2.0,label='Public')
		plt.xlabel('Phase')
		plt.ylabel('Number of Supernova')
		plt.legend()
		plt.title('Number of Supernova by Phase')
		plt.savefig(os.path.join(self.base_dir,'phase_number_histogram.png'))
		plt.close()



##########################################
# Note this next section of code is
# solely to run in a parallel
# fashion. To run it in a single thread/
# process, just create an Abase object
# and call the run_everything() method
##########################################

def white_noise_single(source,target):
	"""
	Helper function to be able to make the white noise plots in parallel.

	:source: (str) choose from snmc.Parameters.sources
	:target: (str) choose from snmc.Parameters.sources
	"""
	params = creator.Parameters()
	base_object = Abase()
	for phase in params.phases:
		for model_mode in params.model_modes:
			base_object.white_noise(source,target,phase,model_mode)

def make_white_noise_parallel():
	"""
	Method to make white noise residual plots in a parallelized fashion using
	processes.
	"""
	processes = []
	params = creator.Parameters()
	for source in params.main_sources:
		for target in params.main_sources:
			p = Process(target=white_noise_single,args=(source,target,))
			p.start()
			processes.append(p)
	for p in processes:
		p.join()

def parallel_components_single(model_mode,source):
	"""
	Helper function to be able to make the web component pages in parallel.

	:model_mode: (str) choose from creator.Parameters.model_sources
	:source: (str) choose from snmc.Parameters.sources
	"""
	base_object = Abase()
	base_object.web_components(model_mode,source)

def make_components_parallel():
	"""
	Method to make the component plots in a parallelized fashion using processes
	"""
	params = creator.Parameters()
	processes = []
	for model_mode in params.model_modes:
		for source in params.main_sources:
			p = Process(target=parallel_components_single\
				,args=(model_mode,source,))
			p.start()
			processes.append(p)
	for p in processes:
		p.join()

def parallel_gallery_single(stat_mode,model_mode):
	"""
	Helper function to be able to make a web gallery page in parallel.

	:stat_mode: (str) choose from ['chi','rms']
	:model_mode: (str) choose from creator.Parameters.model_modes
	"""
	params = creator.Parameters()
	base_object = Abase()
	for source in params.main_sources:
		for target in params.main_sources:
			base_object.gallery_maker(stat_mode,model_mode,source,target)

def make_gallery_parallel():
	"""
	Method to make the web galleries in a parallelized fashion using processes.
	"""
	processes = []
	for stat_mode in ['chi','rms']:
		for model_mode in ['PCA','emFA']:
			p = Process(target=parallel_gallery_single\
				,args=(stat_mode,model_mode,))
			p.start()
			processes.append(p)
	for p in processes:
		p.join()

def parallel_variance_single(model_mode,source):
	"""
	Helper function to be able to make the explained variance curves
	(individual) in parallel.

	Note: in the target for loop, any combination of strings from
		snmc.Parameters.sources are valid to use in the list.

	:model_mode: (str) choose from creator.Parameters.model_modes
	:source: (str) choose from snmc.Parameters.sources
	"""
	base_object = Abase()
	params = creator.Parameters()
	for target in params.main_sources:
		base_object.explained_variance(model_mode,source,target)

def make_variance_parallel():
	"""
	Method to make the variance plots in a parallelized fashion using processes.

	Note: in the source for loop, any combination of strings from
		snmc.Parameters.sources are valid to use in the list.
	"""
	processes = []
	params = creator.Parameters()
	for model_mode in params.model_modes:
		for source in params.main_sources:
			p = Process(target=parallel_variance_single\
				,args=(model_mode,source,))
			p.start()
			processes.append(p)
	for p in processes:
		p.join()

def create_web_gallery(print_progress=True):
	"""
	Create all of the web gallery in a parallelized fashion.

	:print_progress: (bool) (optional) whether to print progress through the
						various parts of building the site
	"""
	if print_progress:
		print('doing the components')
	make_components_parallel()
	if print_progress:
		print('doing the gallery')
	make_gallery_parallel()
	if print_progress:
		print('doing the variance')
	make_variance_parallel()

##########################################
# End of parallelized code section
##########################################

# if __name__ == '__main__':
# 	create_web_gallery()
# 	make_white_noise_parallel()
# 	base = Abase()
# 	base.group_explained_variance_plots()
# 	base.pub_comp_breakdown()