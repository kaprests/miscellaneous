import numpy as np
from matplotlib import pyplot as plt


bounded_treshold = 2 # Assume unbounded if one term has modulus greater than this value
max_iter = 100 # Max iterations for trying to reach treshold value, assume bounded if this limit is reached
z0 = 0 # first term of the zn sequence
resolution = 1000 # Number of points on axes - resolution of final figure

# Define our subset of the complex plane
x = np.linspace(-1.5,0.5, resolution) # range of real values
y = np.linspace(-1,1, resolution) # range of imaginary values
Re = np.tile(x, (resolution, 1)) # Real values matrix
Im = np.tile(y, (resolution, 1)).T * 1j # Imaginary val matrix
Z = Re + Im # Matrix with all complexnumbers in our subset of the complex plane

iters = np.empty((resolution, resolution)) # Array for storing number of iterations for each c in Z
# iter val of max_iter means we assume bounded, less than max_iter means assume unbounded


def zn(z, c):
    return z**2 + c


def get_modulus(c):
    z = z0
    for i in range(max_iter):
        num_iter = i + 1
        z = zn(z, c)
        modulus_z = abs(z)
        if modulus_z > bounded_treshold:
            # Assume bounded
            break
    return num_iter


for i, row in enumerate(Z):
    for j, c in enumerate(row):
        iters[i][j] = get_modulus(c)

plt.imshow(iters)
#plt.savefig('mandelbrotfig.png')
plt.show()
