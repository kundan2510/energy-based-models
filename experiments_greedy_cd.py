import os

dtypes = ['circle', 'spiral', 'quadratic', 'linear', 'four_circle', 'conc_circle']
layers = [1,2,3]
hidden_dim = [10]
num_samples = [100]

command = "THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32 python greedy_cd.py "

for dt in dtypes:
	for l in layers:
		for n in hidden_dim:
			for ns in num_samples:
				out_fol_name = "greedy_cd_{}_{}_{}_{}".format( dt, n, l, ns)
				flags = "-n {} -l {} -d {} -s {} -o {}".format(n, l, dt, ns, out_fol_name)
				os.system(command+flags)
				print "Done {}".format(out_fol_name)
