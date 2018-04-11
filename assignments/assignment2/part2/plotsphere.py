import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from mpl_toolkits.mplot3d import Axes3D

data = np.genfromtxt('output.csv', delimiter=',', skip_header=1, names=['x', 'y', 'z'])

fig = plt.figure()
ax1 = Axes3D(fig)

ax1.scatter(data['x'], data['y'], data['z'], color='black', marker='.')

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

plt.show()