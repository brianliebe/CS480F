import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from mpl_toolkits.mplot3d import Axes3D

data = np.genfromtxt('output.csv', delimiter=',', skip_header=1, names=['x', 'y', 'z'])
x = data['x']
y = data['y']
z = data['z']

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data['x'], data['y'], data['z'], color='black', marker='.')

ax.set_xlabel('Dimension')
ax.set_ylabel('Distance from Center')
ax.set_zlabel('Percentage of Points')

plt.show()