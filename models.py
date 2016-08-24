import pickle

class Model:
	def __init__(self, name=""):
		self.name = name
		self.layers = []
		self.params = []
		self.other_updates = {}
	
	def add_layer(self,layer):
		self.layers.append(layer)
		for p in layer.params:
			self.params.append(p)

		if hasattr(layer, 'other_updates'):
			for y in layer.other_updates:
				self.other_updates[y[0]]=y[1]

	def print_layers(self):
		for layer in self.layers:
			print layer.name

	def get_params(self):
		return self.params

	def print_params(self):
		for p in self.params:
			print p.name

	def save_params(self, file_name):
		params = {}
		for p in self.params:
			params[p.name] = p.get_value()
		pickle.dump(params, open(file_name, 'wb'))

	def load_params(self, file_name):
		params = pickle.load(open(file_name, 'rb'))
		for p in self.params:
			p.set_value(params[p.name])
