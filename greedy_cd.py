import numpy
import matplotlib
matplotlib.use("Agg")

from models import *
from layers import FC, WrapperLayer
import theano
import theano.tensor as T
import lasagne
import random

from matplotlib.pyplot import cm
import matplotlib.pyplot as plt

numpy.random.seed(seed = 123)

def get_data(num_examples, dtype='spiral'):
	"dtype : 'circle', 'spiral', 'quadratic', linear"
	if dtype == "circle":
		a = numpy.linspace(0,2.*numpy.pi, num=num_examples)
		x = numpy.sin(a)
		y = numpy.cos(a)
	elif dtype == "spiral":
		a = numpy.linspace(0.,2.*numpy.pi, num=num_examples)
		norm_ = numpy.linspace(0.,1., num=num_examples)
		x = numpy.sin(a)*norm_
		y = numpy.cos(a)*norm_
	else:
		raise Exception("Not Implemented {} data yet!!".format(dtype))

	data = numpy.concatenate((x[:,None], y[:, None]), axis = 1).astype('float32')
	return data




def floatX(num):
	if theano.config.floatX == 'float32':
		return numpy.float32(num)
	else:
		raise Exception("{} type not supported".format(theano.config.floatX))

def plot_fn_val(fn, rng, points=None, name = ""):
	"rng will be a list of tuples, specifiying range for x and y"
	(x_beg, x_end), (y_beg, y_end) = rng
	X, Y = numpy.mgrid[x_beg:x_end:30j, y_beg:y_end:30j].astype('float32')


	fno = fn(numpy.concatenate((X.reshape((1,-1)), Y.reshape((1,-1))), axis = 0).T).reshape(X.shape+(2,))
	# print fno.shape
	U, V = fno.transpose((2,0,1))

	# print U.shape, V.shape

	speed = numpy.sqrt(U**2 + V**2)

	UN = U/speed
	VN = V/speed

	if points is not None:
		plt.scatter(points[:,0], points[:,1])

	plt.quiver(X, Y, UN, VN,        # data
           U,                   # colour the arrows based on this array
           cmap=cm.winter,     # colour map
           headlength=3,
           clim = [0.,1.]) 

	plt.colorbar()                  # adds the colour bar

	plt.title('Energy Gradient plot')
	plt.savefig('gradient_with_point_{}_circle1.jpg'.format(name)) 
	plt.clf()

def plot_data(X, name):
	plt.scatter(X[:,0], X[:,1], '.')
	plt.savefig("data_{}.jpg".format(name))


def create_energy_model(input_size, hidden_dim, hidden_layers, model, X, name = ""):

	mlp_input = FC(input_size, hidden_dim, WrapperLayer(X), name = name+".input")
	model.add_layer(mlp_input)

	last_layer = mlp_input

	for i in range(hidden_layers):
		fc = FC(hidden_dim, hidden_dim, WrapperLayer(T.nnet.sigmoid(last_layer.output())), name = name+".fc_{}".format(i+1))
		model.add_layer(fc)
		last_layer = fc

	energy = FC(hidden_dim, 1, WrapperLayer(T.nnet.sigmoid(last_layer.output())), name = name + ".output")

	return energy.output()


model = Model(name="preliminary")

params = model.get_params()

X = T.matrix('X')

eta = T.scalar('eta')

energy = create_energy_model(2, 1024, 2, model, X, name="preliminary")

energy_sum = T.sum(energy)

energy_grad_X = T.grad(energy_sum, wrt=X, disconnected_inputs='warn')

energy_grad_theta = T.grad(energy_sum, params)
neg_energy_grad_theta = [floatX(-1.)*g for g in energy_grad_theta]

negative_examples = X - eta*energy_grad_X

update_pos = lasagne.updates.adam(energy_grad_theta, params, 0.0001)
update_neg = lasagne.updates.adam(neg_energy_grad_theta, params, 0.0001)

get_neg_examples = theano.function([X, eta], negative_examples)

train_up = theano.function([X], energy_sum, updates = update_pos)
train_down = theano.function([X], energy_sum, updates = update_neg)

get_energy = theano.function([X], energy_sum)
get_energy_grad = theano.function([X], energy_grad_X)

def train_batch(x, eps):
	pos_examples = x
	n_x = get_neg_examples(x, floatX(eps))
	en_x = train_up(x)
	en_n_x = train_down(n_x)
	return en_x, en_n_x
	
# plot_fn_val(get_energy_grad, [(-3,3),(-3,3)])
X = get_data(1000, dtype="circle")



E_x = []

eps = 0.001
for i in range(1000000):
	batch = X[random.sample(xrange(1000), 10)]
	en_x, en_n_x = train_batch(batch, eps)
	E_x.append(en_x)
	if (i % 100) == 0:
		print "Energy for X in iter {} : {}, X_neg : {}".format(i, en_x, en_n_x)
		plt.plot(numpy.arange(len(E_x)), numpy.asarray(E_x), ".-")
		plt.savefig("Energy_diagram_circle1.jpg")
		plt.clf()

	if (i % 5000) == 0:
		eps = eps*0.75
		plot_fn_val(get_energy_grad, [(-1.5,1.5),(-1.5,1.5)], points=X, name = "iter_{}".format(i) )

	











