import matplotlib
matplotlib.use("Agg")

from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import numpy as np

Y, X = np.mgrid[-3:3:15j, -3:3:15j].astype('float32')
# U = -1 - np.cos(X**2 + Y)
# V = 1 + X - Y
# speed = np.sqrt(U**2 + V**2)
# UN = U/speed
# VN = V/speed


# Y, X = np.mgrid[-3:3:15j, -3:3:15j]
# U = -1 - np.cos(X**2 + Y)
# V = 1 + X - Y
# U = X + 0.000001
# V = Y + 0.000001
U, V = np.random.random((2,15,15))
speed = np.sqrt(U**2 + V**2)
UN = U/speed
VN = V/speed

# plot1 = plt.figure()
plt.quiver(X, Y, UN, VN,        # data
           U,                   # colour the arrows based on this array
           cmap=cm.seismic,     # colour map
           headlength=7)        # length of the arrows

plt.colorbar()                  # adds the colour bar

plt.title('Quive Plot, Dynamic Colours')
plt.savefig('test_quiver.jpg') 
