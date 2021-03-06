timport numpy
import matplotlib
matplotlib.use("Agg")

from models import *
from layers import FC, WrapperLayer
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
add_arg("-iters","--num_iters", default=15000, type = int, help= "Number of iterations to run")
add_arg("-save", "--save_after", default=1500, type = int, help= "Saving result after")

add_arg("-o","--output_dir", required=True, help= "Output directory path to save the results")

args = parser.parse_args()

print_args(args)

if not os.path.isdir(args.output_dir):
	os.makedirs(args.output_dir)

numpy.random.seed(seed = 123)

def get_data(num_examples, dtype='spiral'):
	"dtype : 'circle', 'spiral', 'quadratic', linear"
	if dtype == "circle":
		a = numpy.linspace(0,2.*numpy.pi, num=num_examples)
		x = numpy.sin(a)
		y = numpy.cos(a)
	elif dtype == "conc_circle":
		a = numpy.linspace(0,4.*numpy.pi, num=num_examples)
		x = numpy.concatenate((numpy.sin(a[:num_examples//2]), 2*numpy.sin(a[num_examples//2:])))
		y = numpy.concatenate((numpy.cos(a[:num_examples//2]), 2*numpy.cos(a[num_examples//2:])))
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

	if points is not None:
		plt.scatter(points[:,0], points[:,1])

	plt.quiver(X, Y, U, V,        # data
           color='DarkRed'
           )

	# plt.colorbar()                  # adds the colour bar

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
	plt.savefig(os.path.join(args.output_dir, "data_{}.jpg".format(name)))


def create_energy_model(input_size, hidden_dim, hidden_layers, model, X, name = ""):

	mlp_input = FC(input_size, hidden_dim, WrapperLayer(X), name = name+".input")
	model.add_layer(mlp_input)

	last_layer = mlp_input

	for i in range(hidden_layers):
		fc = FC(hidden_dim, hidden_dim, WrapperLayer(T.nnet.elu(last_layer.output())), name = name+".fc_{}".format(i+1))
		model.add_layer(fc)
		last_layer = fc

	energy = FC(hidden_dim, 1, WrapperLayer(T.nnet.elu(last_layer.output())), name = name + ".output")

	return energy.output()


model = Model(name="preliminary")

params = model.get_params()

X = T.matrix('X')

eta = T.scalar('eta')

energy = create_energy_model(2, 1024, 2, model, X, name="preliminary")

energy_sum = T.sum(energy)

energy_grad_X = T.grad(energy_sum, wrt=X, disconnected_inputs='warn')

energy_grad_theta = T.grad(energy_sum, params)
energy_grad_theta = [T.clip(g, floatX(-1.0), floatX(1.0)) for g in energy_grad_theta]

neg_energy_grad_theta = [floatX(-1.)*g for g in energy_grad_theta]

negative_examples = X - eta*energy_grad_X

update_pos = lasagne.updates.adam(energy_grad_theta, params, 0.001)
update_neg = lasagne.updates.adam(neg_energy_grad_theta, params, 0.001)

get_neg_examples = theano.function([X, eta], negative_examples)

train_up = theano.function([X], energy_sum, updates = update_pos)
train_down = theano.function([X], energy_sum, updates = update_neg)

get_energy = theano.function([X], energy_sum)
get_energy_grad = theano.function([X], floatX(-1)*energy_grad_X)

def train_batch(x, eps):
	pos_examples = x
	n_x = get_neg_examples(x, floatX(eps))
	en_x = train_up(x)
	en_n_x = train_down(n_x)
	return en_x, en_n_x
	
# plot_fn_val(get_energy_grad, [(-3,3),(-3,3)])



E_x = []

X = get_data(args.num_samples, dtype=args.data_type)

eps = 0.0001
for i in range(1000000):
	batch_ind = random.sample(range(0, args.num_samples), 100)
	batch = X[batch_ind]
	en_x, en_n_x = train_batch(batch, eps)
	E_x.append(en_x)
	if (i % 100) == 0:
		print "Energy for X in iter {} : {}, X_neg : {}".format(i, en_x, en_n_x)
		plt.plot(numpy.arange(len(E_x)), numpy.asarray(E_x), ".-")
		plt.savefig(os.path.join(args.output_dir,"Energy_diagram_{}.jpg".format(args.data_type)))
		plt.clf()

	if (i % args.save_after) == 0:
		plot_fn_val(get_energy_grad, [(-3.5,3.5),(-3.5,3.5)], points=X, name = "iter_{}_{}".format(args.data_type, i) )

	
	if i % 1000:
		eps = eps*0.75

	if i == args.num_iters:
		exit()











