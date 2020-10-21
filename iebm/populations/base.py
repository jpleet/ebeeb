import numpy as numpy

class Population():
	""" Base population class. Most code are in subclasses

	Example
	-------
	TO DO: example
	"""

	def __init__(self, name, init_size, implicit_capacity=None):
		""" Constructor for base population
		
		Parameters
		----------

		name : str
			unique name to indentify population from other populations
		init_size : int
			starting size of the population
		implicit_capacity : int or None
			max size the population can reach. if None, no limit

		"""

		self.name = name
		self.size = init_size
		self.implicit_capacity = implicit_capacity

