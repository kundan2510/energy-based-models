import numpy
import matplotlib
matplotlib.use("Agg")

from models import *
from layers import FC, WrapperLayer, ExponentialCost
import theano
import theano.tensor as T
import lasagne
import random
from generic_utils import *


from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import argparse

T.nnet.elu = lambda x: T.switch(x >= floatX(0.), x, T.exp(x) - floatX(1.))


parser = argparse.ArgumentParser(description="Generating samples using iterative methods")
add_arg = parser.add_argument

add_arg("-l","--hidden_layers", default=2, type = int, help= "number of hidden layers")
add_arg("-n","--hidden_units", default=100, type = int, help= "Number of hidden units/neurons in each hidden layer")
add_arg("-d","--data_type", default="spiral", help= "Type of data")
add_arg("-s","--num_samples", default=100, type = int, help= "Number of samples")
add_arg("-co","--num_components", default=2, type = int, help= "Number of samples")
add_arg("-iters","--num_iters", default=2000, type = int, help= "Number of iterations to run")
add_arg("-save", "--save_after", default=200, type = int, help= "Saving result after")
add_arg("-t", "--temp", default=1., type = float, help= "Temperature")

add_arg("-o","--output_dir", required=True, help= "Output directory path to save the results")

args = parser.parse_args()

print_args(args)

if not os.path.isdir(args.output_dir):
	os.makedirs(args.output_dir)


numpy.random.seed(seed = 44664)

def get_data(num_examples, dtype='spiral'):
	"dtype : 'circle', 'spiral', 'quadratic', 'linear', 'four_circle', 'conc_circle' "
	if dtype == "circle":
		a = numpy.linspace(0,2.*numpy.pi, num=num_examples)
		x = numpy.sin(a)
		y = numpy.cos(a)
	elif dtype == "conc_circle":
		a = numpy.linspace(0,4.*numpy.pi, num=num_examples)
		x = numpy.concatenate((numpy.sin(a[:num_examples//2]), 2.6*numpy.sin(a[num_examples//2:])))
		y = numpy.concatenate((numpy.cos(a[:num_examples//2]), 2.6*numpy.cos(a[num_examples//2:])))
	elif dtype == "four_circle":
		a = numpy.linspace(0,8.*numpy.pi, num=num_examples)
		x = 0.25*numpy.sin(a)
		y = 0.25*numpy.cos(a)

		x[:num_examples//4] += 0.5; y[:num_examples//4] += 0.5
		x[num_examples//4:num_examples//2] += 0.5; y[num_examples//4:num_examples//2] -= 0.5
		x[num_examples//2:3*num_examples//4] -= 0.5; y[num_examples//2:3*num_examples//4] += 0.5
		x[3*num_examples//4:] -= 0.5; y[3*num_examples//4:] -= 0.5
	elif dtype == "spiral":
		a = numpy.linspace(0.,8.*numpy.pi, num=num_examples)
		norm_ = numpy.linspace(0.,3., num=num_examples)
		x = numpy.sin(a)*norm_
		y = numpy.cos(a)*norm_
	elif dtype == "linear":
		a = numpy.linspace(-1.5,1.5, num=num_examples)
		x = a
		y = a
	elif dtype == "quadratic":
		a = numpy.linspace(-1.5,1.5, num=num_examples)
		x = a
		y = a**2 - 1.5
	elif dtype == "concave_quadratic":
		a = numpy.linspace(-1.5,1.5, num=num_examples)
		x = a
		y = -a**2 + 1.5
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
	X, Y = numpy.mgrid[x_beg:x_end:40j, y_beg:y_end:40j].astype('float32')


	fno = fn(numpy.concatenate((X.reshape((1,-1)), Y.reshape((1,-1))), axis = 0).T).reshape(X.shape+(2,))
	# print fno.shape
	U, V = fno.transpose((2,0,1))

	# print U.shape, V.shape

	speed = numpy.sqrt(U**2 + V**2)

	UN = U/speed
	VN = V/speed

	U_ = U/speed.max()
	V_ = V/speed.max()

	if points is not None:
		plt.scatter(points[:,0], points[:,1])

	plt.quiver(X, Y, U_, V_,        # data
           color='DarkRed'
           )

	#plt.colorbar()                  # adds the colour bar

	plt.title('Energy Gradient plot')
	plt.savefig(os.path.join(args.output_dir,'gradient_with_point_{}.jpg'.format(name)))
	plt.clf()

	if points is not None:
		plt.scatter(points[:,0], points[:,1])
		
	plt.quiver(X, Y, UN, VN,        # data
           color='DarkRed'
           )

	#plt.colorbar()                  # adds the colour bar

	plt.title('Energy Gradient plot')
	plt.savefig(os.path.join(args.output_dir,'gradient_with_point_{}_direction.jpg'.format(name)))
	plt.clf()

def plot_data(X, name):
	plt.scatter(X[:,0], X[:,1], '.')
	plt.savefig("data_{}.jpg".format(name))


def create_energy_model(input_size, hidden_dim, hidden_layers, num_components, model, X, name = ""):

	mlp_input = FC(input_size, hidden_dim, WrapperLayer(X), name = name+".input")
	model.add_layer(mlp_input)

	last_layer = mlp_input

	for i in range(hidden_layers):
		fc = FC(hidden_dim, hidden_dim, WrapperLayer(T.nnet.elu(last_layer.output())), name = name+".fc_{}".format(i+1))
		model.add_layer(fc)
		last_layer = fc

	fc = FC(hidden_dim, hidden_dim, last_layer, name = name+".fc_output")
	model.add_layer(fc)
	last_layer = fc

	energy = ExponentialCost(hidden_dim, num_components, last_layer, temp= args.temp, name = name+".exp_cost")

	return energy.output()


model = Model(name="preliminary")

params = model.get_params()

X = T.matrix('X')

eta = T.scalar('eta')

energy = create_energy_model(2, args.hidden_units, args.hidden_layers, args.num_components, model, X, name="preliminary")

energy_grad_X = T.grad(energy, wrt=X, disconnected_inputs='warn')

energy_grad_theta = T.grad(energy, params)
energy_grad_theta = [T.clip(g, floatX(-1.0), floatX(1.0)) for g in energy_grad_theta]

updates = lasagne.updates.adam(energy_grad_theta, params, 0.005)

train_fn = theano.function([X], energy, updates = updates)

get_energy = theano.function([X], energy)
get_energy_grad = theano.function([X], floatX(-1.)*energy_grad_X)

def train_batch(x, eps):
	return train_fn(x)

# plot_fn_val(get_energy_grad, [(-3,3),(-3,3)])



E_x = []

X = get_data(args.num_samples, dtype=args.data_type)

eps = 0.0001
for i in range(1000000):
	batch_ind = random.sample(range(0, args.num_samples), 100)
	batch = X[batch_ind]
	en_x = train_batch(batch, eps)
	E_x.append(en_x)
	if (i % 100) == 0:
		print "Energy for X in iter {} : {}".format(i, en_x)
		plt.plot(numpy.arange(len(E_x)), numpy.asarray(E_x), ".-")
		plt.savefig(os.path.join(args.output_dir,"Energy_diagram_{}_{}.jpg".format(args.data_type, args.temp)))
		plt.clf()

	if (i % args.save_after) == 0:
		eps = eps*0.75
		plot_fn_val(get_energy_grad, [(-3.5,3.5),(-3.5,3.5)], points=X, name = "iter_{}_{}_{}".format(args.data_type, args.temp, i) )

	if i == args.num_iters:
		exit()












