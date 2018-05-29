# High Performance Computing (CS480F/CS580F)

## Project 1 (C++)

A parallel 'k-NN' problem, in which you find the 'k'-nearest neighbors given a set of points over any number of dimensions. 
This problem was implemented using a k-d tree structure, where each layer of the tree is a split on an alternating dimension. 

## Project 2 (C++ / Python)

### Part 1 (C++)

Computation of 4-dimension line lengths using SIMD architecture to achieve speed-ups against the normal sequential version.

### Part 2 (C++ / Python)

As the dimensionality of a sphere increases, the volume of the sphere tends to lie near the surface. 
This program proves this fact by first quickly and randomly generating points within the sphere using a complex equation.
The points are then plotted with matplotlib using a surface and scatter plot to show that the points tend to lie near the surface in higher dimensions. 

By using this equation, speedup was significant compared to the standard "reject" technique of finding points within a N-dimensional unit square and rejecting points outside the circle.
I was able to easily handle around 100 dimensions when the reject technique would take hours at that level. I also implemented OpenMP to increase speedup even more.

## Project 3 (C++ / CUDA)

This project implemented a convolutional neural network (CNN) to recognize hand-written numbers, 0-9 (the MNIST dataset). 
The CNN consisted of several layers, including two dense layers, the softmax layer, and the cross-entropy layer. 
The CNN works by using thousands of weights that are adjusted on each iteration, and then trained to better recognize each letter. 
CUDA was worked into the design of the CNN to help speed up some aspects of the program, primarily for training. 
CUDA works by copying the arrays onto a GPU and quickly generate answers using thousands of threads at once.
