import os

dtypes = ['circle', 'concave_quadratic', 'conc_circle']
layers = [1,2]
hidden_dim = [10]
temp = [1, 0.9, 0.7, 0.5, 0.1]
num_samples = [10000]
num_components = [1, 2, 5, 10, 100]

command = "THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32 python gmm_energy.py "

for dt in dtypes:
	for l in layers:
		for n in hidden_dim:
			for ns in num_samples:
				for t in temp:
					for co in num_components:
						out_fol_name = "/Tmp/kumarkun/gmm_sum_energy/{}/args_{}_{}_{}_{}_{}".format( dt, n, l, ns, t, co)
						flags = "-n {} -l {} -d {} -s {} -o {} -t {} -co {}".format(n, l, dt, ns, out_fol_name, t, co)
						os.system(command+flags)
						print "Done {}".format(out_fol_name)
