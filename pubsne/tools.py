"""
Project: pubsne
Authors:
	kceades
"""

# imports from standard packages
import numpy as np
import time

class Fitter(object):
	"""
	Class object for a fitter: this will generate the coefficients for fitting
	a vector to an arbitrary set of other vectors, or reduce the vector by
	fitting out vectors.
	"""
	def __init__(self):
		"""
		Constructor
		"""
		self.base_vecs = None
		self.inv_vecs = None

	def prep_fit(self,base_vecs):
		"""
		Sets up the inverse matrix with base_vecs for use in fitting.

		:base_vecs: (list of arrays or lists of floats) list of vectors of the 
					same length
		"""
		num_vecs = len(base_vecs)

		coeff_arr = np.zeros((num_vecs,num_vecs))
		for i in range(num_vecs):
			for j in range(i):
				coeff_arr[i][j] = coeff_arr[j][i]
			for j in range(i,num_vecs):
				coeff_arr[i][j] = np.dot(base_vecs[i],base_vecs[j])

		self.inv_vecs = np.linalg.inv(coeff_arr)
		self.base_vecs = base_vecs

	def fit(self,vec):
		"""
		Does the actual fitting and returns the coefficients.

		:vec: (list or array of floats) vector of the same length as all
				vectors in self.base_vecs

		:returns: (array of floats) the coefficients of the fit
		"""
		if self.base_vecs is None or self.inv_vecs is None:
			raise OrderError('Fit','Invalid Order: Run fitter.PrepFit before' \
				+ ' running fitter.Fit')

		final = [np.dot(vec,base_vec) for base_vec in self.base_vecs]

		return np.transpose(np.dot(self.inv_vecs,np.transpose(final)))

	def reduce(self,vecs):
		"""
		Returns the reduced vectors of vecs after being fit to self.base_vecs

		:vecs: (list of arrays or lists of floats) list of the vectors to be
				reduced which are themselves composed of floats

		:returns: (list of arrays or lists of floats) list of the reduced
					vectors that are themselves floats
		"""
		if self.base_vecs is None or self.inv_vecs is None:
			raise KeyError('The fitter has not yet been prepped.')

		r_arr = []
		for vec in vecs:
			coeffs = self.Fit(vec)
			r_arr.append(np.subtract(vec,np.dot(coeffs,self.base_vecs)))

		return r_arr

def exclusive_in(in_list,master_list):
	"""
	Checks if every element in in_list is in masterlist

	:in_list: (list of any type of object) elements
	:master_list: (list of any type of object) elements

	:returns: (bool) True if every element in in_list is in master_list and
				False otherwise
	"""
	if in_list==[]:
		return True
	else:
		for elem in in_list:
			if elem not in master_list:
				return False
		return True

def remove_dash(string):
	""" 
	Removes the dash from the end of the string if there is one there.

	:string: (str) an input string

	:returns: (str) input string with no dash at the end of it
	"""
	if string[-1]=='-':
		return string[:-1]
	else:
		return string

def closest_match(num,num_list):
	"""
	Finds the closest number to num in num_list.

	:num: (float) a number
	:num_list: (array of floats)

	:returns: (float) the number in num_list closest to num
	"""
	diffs = np.abs(np.subtract(num,num_list))
	return num_list[np.argmin(diffs)]

def fit_rms(a,b):
	"""
	Calculates the RMS of the difference between equal length vectors a and b.

	:a: (array or list of floats) vector
	:b: (array or list of floats) vector

	:returns: (float) rms of a-b
	"""
	if len(a)!=len(b):
		raise ValueError('Input vectors have unequal lengths.')
	return rms(np.subtract(a,b))

def rms(a):
	"""
	Calculates the RMS of a vector.

	:a: (array or list of floats) vector

	:returns: (float) rms of a
	"""
	return np.sqrt(np.sum(np.power(a,2))/len(a))

def rootsumsquares(a):
	"""
	Calculates the square root of the sum of the squares of a vector (no mean).

	:a: (array or list of floats) vector

	:returns: (float) rms without the mean part of a
	"""
	return np.sqrt(np.sum(np.power(a,2)))

def fit_chi(fit,data,errs,dof):
	"""
	Calculates the chi-square when fitting data with fit while having
	uncertainties errs and dof degrees of freedom (note fit, data and errs
	all have to be of uniform length).

	:fit: (list or array of floats) the fit to the data
	:data: (list or array of floats) the data
	:errs: (list or array of floats) the error (std) on the data
	:dof: (int) the number of degrees of freedom

	:returns: (float) the chi-square per degree of freedom of the fit
	"""
	if len(fit)!=len(data) or len(data)!=len(errs):
		raise ValueError('Input vectors have unequal lengths.')
	squared_err = np.power(np.subtract(fit,data),2)
	return np.sum(np.divide(squared_err,np.power(errs,2)))/dof

def time_it(func,*args):
	"""
	Wrapper for a function to time it's execution.

	:func: (function) the function to time
	:*args: (optional) arguments to the function

	:returns: (float) the time of execution in minutes, rounded to 1 decimal
	"""
	t0 = time.time()
	func(*args)
	t1 = time.time()
	diff = t1-t0
	return np.round(diff/60,1)