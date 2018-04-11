from mpl_toolkits.mplot3d import *
import matplotlib.pyplot as plt
import numpy as np
from random import random, seed
from matplotlib import cm


fig = plt.figure()
ax = fig.gca(projection='3d')

data = np.genfromtxt('output.csv', delimiter=',', skip_header=1, names=['x', 'y', 'z'])
x = data['x']
y = data['y']
z = data['z']
# ax.scatter(x, y, z, color='black', marker='.')

ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')

ax.set_xlabel('Dimension')
ax.set_ylabel('Distance from Center')
ax.set_zlabel('Percentage of Points')

plt.show()