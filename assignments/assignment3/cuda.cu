#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <random>
#include <algorithm>
#include <vector>
#include <unistd.h>
#include <ctime>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

const int IMAGE_DIM = 28; 
const int FCL1_DIM = 1024; 
const int FCL2_DIM = 10; 

void switchEndian(int &ui)
{
    ui = (ui >> 24) | ((ui<<8) & 0x00FF0000) | ((ui>>8) & 0x0000FF00) | (ui << 24);
}

float ***read_image_file(std::string filename, int &magic_number, int &number_of_items, int &rows, int &columns) 
{
	std::ifstream infile;
	infile.open(filename.c_str(), std::ios::binary);
	infile.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
	infile.read(reinterpret_cast<char*>(&number_of_items), sizeof(number_of_items));
	infile.read(reinterpret_cast<char*>(&rows), sizeof(rows));
	infile.read(reinterpret_cast<char*>(&columns), sizeof(columns));

	switchEndian(magic_number);
	switchEndian(number_of_items);
	switchEndian(rows);
	switchEndian(columns);

	float ***pixels = new float**[number_of_items];
	for (int i = 0; i < number_of_items; i++)
	{
		pixels[i] = new float*[rows];
		for (int r = 0; r < rows; r++)
		{
			pixels[i][r] = new float[columns];
			for (int c = 0; c < columns; c++)
			{
				unsigned char temp;
				infile.read(reinterpret_cast<char*>(&temp), 1);
				pixels[i][r][c] = ((int)temp / 127.5) - 1.0;
			}
		}
	}
	return pixels;
}

int *read_label_file(std::string filename, int &magic_number, int &number_of_items)
{
	std::ifstream infile;
	infile.open(filename.c_str(), std::ios::binary);
	infile.read(reinterpret_cast<char*>(&magic_number), 4);
	infile.read(reinterpret_cast<char*>(&number_of_items), 4);

	switchEndian(magic_number);
	switchEndian(number_of_items);

	int *labels = new int[number_of_items];
	for (int i = 0; i < number_of_items; i++) 
	{
		char temp;
		infile.read(reinterpret_cast<char*>(&temp), 1);
		labels[i] = temp;
	}
	return labels;
}

float *first_dense_layer(float **image, float ***weights, float *bias)
{
	srand(123);

	float *sums = new float[FCL1_DIM];
	for (int i = 0; i < FCL1_DIM; i++)
	{
		float sum = 0;
		int random_num = rand() % 1000;
		if (random_num < 4)
		{
			sums[i] = 0;
			continue;
		}
		for (int r = 0; r < IMAGE_DIM; r++)
		{
			for (int c = 0; c < IMAGE_DIM; c++)
			{
				sum += weights[i][r][c] * image[r][c];
			}
		}
		sum += bias[i];
		sums[i] = (sum > 0) ? sum : 0;
	}
	return sums;
}

float *second_dense_layer(float *layer, float **weights, float *bias)
{
	float *sums = new float[FCL2_DIM];
	for (int i = 0; i < FCL2_DIM; i++) // 10
	{
		float sum = 0;
		for (int j = 0; j < FCL1_DIM; j++) // 1024
		{
			sum += weights[i][j] * layer[j];
		}
		sum += bias[i];
		sums[i] = sum;
	}
	return sums;
}

float *soft_max(float *layer)
{
	float *output = new float[FCL2_DIM];
	float sum = 0;
	float max = layer[0];
	for (int i = 0; i < FCL2_DIM; i++)
	{
		max = (max > layer[i]) ? max : layer[i];
	}
	for (int i = 0; i < FCL2_DIM; i++)
	{
		output[i] = exp(layer[i] - max);
		sum += output[i];
	}
	for (int i = 0; i < FCL2_DIM; i++)
	{
		output[i] = output[i] / sum;
	}
	return output;
}

float cross_entropy(float *soft_maxes, int label)
{
	return -1 * log(soft_maxes[label]);
}

void soft_max_backprop(float *downstream, float *upstream, float *sm_output)
{
	for (int i = 0; i < FCL2_DIM; i++) 
	{
		downstream[i] = 0;
		for (int j = 0; j < FCL2_DIM; j++)
		{
			if (i == j) 
			{
				downstream[i] += upstream[j] * (sm_output[j] * (1 - sm_output[i]));
			}
			else 
			{
				downstream[i] += upstream[j] * (-sm_output[i] * sm_output[j]);
			}
		}
	}
}

void fcl_2_backprop(float *downstream, float *upstream, float *fcl_1_output, float *bias_deriv, float **weights, float **weight_deriv, float batch_size)
{
	for (int i = 0; i < FCL2_DIM; i++)
	{
		for (int j = 0; j < FCL1_DIM; j++)
		{
			downstream[j] = 0;
		}
		for (int j = 0; j < FCL1_DIM; j++)
		{
			downstream[j] += upstream[i] * weights[i][j];
			weight_deriv[i][j] += (upstream[i] * fcl_1_output[j]) / batch_size;
		}
		bias_deriv[i] = upstream[i] / batch_size;
	}
}

void fcl_1_backprop(float *downstream, float *upstream, float **image_output, float *bias_deriv, float ***weights, float ***weight_deriv, float batch_size)
{
	for (int i = 0; i < FCL1_DIM; i++)
	{
		for (int a = 0; a < IMAGE_DIM; a++)
		{
			for (int b = 0; b < IMAGE_DIM; b++)
			{
				weight_deriv[i][a][b] += (upstream[i] * image_output[a][b]) / batch_size;
			}
		}
		bias_deriv[i] = upstream[i] / batch_size;
	}
}

void fcl_1_update(float ***weights, float ***weight_deriv, float *bias, float *bias_deriv, float rate)
{
	for (int i = 0; i < FCL1_DIM; i++)
	{
		for (int a = 0; a < IMAGE_DIM; a++)
		{
			for (int b = 0; b < IMAGE_DIM; b++)
			{
				weights[i][a][b] -= rate * weight_deriv[i][a][b];
				weight_deriv[i][a][b] = 0;
			}
		}
		bias[i] -= rate * bias_deriv[i];
		bias_deriv[i] = 0;
	}
}

__global__ void cuda_fcl1_update(float *weights, float *weight_deriv, float *bias, float *bias_deriv, float rate)
{
	int n = 28 * 28 * 1024;
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if (id > n) return;

	weights[id] -= rate * weight_deriv[id];
	weight_deriv[id] = 0;
	if (id % (28 * 28) == 0) 
	{
		bias[id / (28 * 28)] -= rate * bias_deriv[id / (28 * 28)];
		bias_deriv[id / (28 * 28)] = 0;
	}
}

void fcl_2_update(float **weights, float **weight_deriv, float *bias, float *bias_deriv, float rate)
{
	for (int i = 0; i < FCL2_DIM; i++)
	{
		for (int j = 0; j < FCL1_DIM; j++)
		{
			weights[i][j] -= rate * weight_deriv[i][j];
			weight_deriv[i][j] = 0;
		}
		bias[i] -= rate * bias_deriv[i];
		bias_deriv[i] = 0;
	}
}

__global__ void cuda_fcl2_update(float *weights, float *weight_deriv, float *bias, float *bias_deriv, float rate)
{
	int n = 1024 * 10;
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if (id > n) return;

	weights[id] -= rate * weight_deriv[id];
	weight_deriv[id] = 0;
	if (id % (1024) == 0)
	{
		bias[id / 1024] -= rate * bias_deriv[id / 1024];
		bias_deriv[id / 1024] = 0;
	}
}

float ***create_weights(int x, int y, int z)
{
	std::default_random_engine eng;
	eng.seed(123);
	std::normal_distribution<float> init;

	float ***ret = new float**[x];
	for (int i = 0; i < x; i++) // 1024
	{
		ret[i] = new float*[y];
		for (int j = 0; j < y; j++) // 28
		{
			ret[i][j] = new float[z];
			for (int k = 0; k < z; k++) // 28
			{
				ret[i][j][k] = init(eng) / sqrt(x);
			}
		}
	}
	return ret;
}

float **create_weights(int x, int y)
{
	std::default_random_engine eng;
	eng.seed(456);
	std::normal_distribution<float> init;
	float **ret = new float*[x];

	for (int i = 0; i < x; i++) // 10
	{
		ret[i] = new float[y];
		for (int j = 0; j < y; j++) // 1024
		{
			ret[i][j] = init(eng) / sqrt(x);
		}
	}
	return ret;
}

float ***create_weight_deriv(int a, int b, int c)
{
	float ***ret = new float**[a];
	for (int i = 0; i < a; i++) // 1024
	{
		ret[i] = new float*[b];
		for (int j = 0; j < b; j++) // 28
		{
			ret[i][j] = new float[c];
			for (int k = 0; k < c; k++) // 28
			{
				ret[i][j][k] = 0;
			}
		}
	}
	return ret;
}

float **create_weight_deriv(int a, int b)
{
	float **ret = new float*[a];
	for (int i = 0; i < a; i++) // 10
	{
		ret[i] = new float[b];
		for (int j = 0; j < b; j++) // 1024
		{
			ret[i][j] = 0;
		}
	}
	return ret;
}

int main(int argc, char **argv)
{
	if (argc != 3)
	{
		printf("USAGE: ./prog3 <image file> <label file>\n");
		return 0;
	}

	int magic_number_label, magic_number_image, number_of_labels, number_of_images, rows, columns;

	// Read in the labels and images and store them
	float ***images = read_image_file(std::string(argv[1]), magic_number_image, number_of_images, rows, columns);
	int *labels = read_label_file(std::string(argv[2]), magic_number_label, number_of_labels);
	assert(number_of_labels == number_of_images);

	float ***image_layer_weights = create_weights(FCL1_DIM, IMAGE_DIM, IMAGE_DIM);
	float **connected_layer_weights = create_weights(FCL2_DIM, FCL1_DIM);
	float ***image_weight_deriv = create_weight_deriv(FCL1_DIM, IMAGE_DIM, IMAGE_DIM);
	float **connected_weight_deriv = create_weight_deriv(FCL2_DIM, FCL1_DIM);

	float *fcl_1 = new float[FCL1_DIM]();
	float *fcl_1_bias = new float[FCL1_DIM]();
	float *fcl_1_ds_deriv = new float[FCL1_DIM]();
	float *fcl_1_bias_deriv = new float[FCL1_DIM]();

	float *fcl_2 = new float[FCL2_DIM]();
	float *fcl_2_bias = new float[FCL2_DIM]();
	float *fcl_2_ds_deriv = new float[FCL1_DIM]();
	float *fcl_2_bias_deriv = new float[FCL2_DIM]();

	float *sm_layer = new float[FCL2_DIM]();
	float *sm_ds_deriv = new float[FCL2_DIM]();
	float *ce_ds_deriv = new float[FCL2_DIM]();

	// CUDA variables

	float *d_fcl1_weights, *d_fcl1_weight_deriv, *d_fcl1_bias, *d_fcl1_bias_deriv;
	float *h_fcl1_weights = new float[28*28*1024];
	float *h_fcl1_weight_deriv = new float[28*28*1024];

	float *d_fcl2_weights, *d_fcl2_weight_deriv, *d_fcl2_bias, *d_fcl2_bias_deriv;
	float *h_fcl2_weights = new float[10*1024];
	float *h_fcl2_weight_deriv = new float[10*1024];

	cudaMalloc(&d_fcl1_weights, 28*28*1024*sizeof(float));
	cudaMalloc(&d_fcl1_weight_deriv, 28*28*1024*sizeof(float));
	cudaMalloc(&d_fcl1_bias, 1024*sizeof(float));
	cudaMalloc(&d_fcl1_bias_deriv, 1024*sizeof(float));

	cudaMalloc(&d_fcl2_weights, 10*1024*sizeof(float));
	cudaMalloc(&d_fcl2_weight_deriv, 10*1024*sizeof(float));
	cudaMalloc(&d_fcl2_bias, 10*sizeof(float));
	cudaMalloc(&d_fcl2_bias_deriv, 10*sizeof(float));

	/* TRAINING */
	std::clock_t start = std::clock();
	for (int epoch = 0; epoch < 100; epoch++) 
	{	
		std::clock_t start = std::clock();
		for (int r = 0; r < 100; r++)
		{
			float wrong = 0;
			float right = 0;
			int mini_batch = 100;
			//printf("Starting round: %d\n", r);
			
			for (int image = 0; image < mini_batch; image++)
			{
				int batch_size = mini_batch;
				float rate = mini_batch / 1000.0; // 0.1 for 100 images, 0.01 for 10 images
				
				int label = labels[100*r + image];

				// FCL 1 Forward
				fcl_1 = first_dense_layer(images[100*r + image], image_layer_weights, fcl_1_bias);
				
				// FCL 2 Forward
				fcl_2 = second_dense_layer(fcl_1, connected_layer_weights, fcl_2_bias);
				
				// Softmax Forward
				sm_layer = soft_max(fcl_2);

				// Compute Right/Wrong
				int index = std::distance(sm_layer, std::max_element(sm_layer, sm_layer + FCL2_DIM));
				if (index == label) right++;
				else wrong++;

				// Cross Entropy Forward
				for (int j = 0; j < FCL2_DIM; j++) ce_ds_deriv[j] = 0;
				ce_ds_deriv[label] = -1 / sm_layer[label];

				// Softmax Backprop
				soft_max_backprop(sm_ds_deriv, ce_ds_deriv, sm_layer);

				// FCL 2 Backprop
				fcl_2_backprop(fcl_2_ds_deriv, sm_ds_deriv, fcl_1, fcl_2_bias_deriv, connected_layer_weights, connected_weight_deriv, batch_size);

				// FCL 1 Backprop
				fcl_1_backprop(fcl_1_ds_deriv, fcl_2_ds_deriv, images[100*r + image], fcl_1_bias_deriv, image_layer_weights, image_weight_deriv, batch_size);		

				// FCL 1 Update
				// fcl_1_update(image_layer_weights, image_weight_deriv, fcl_1_bias, fcl_1_bias_deriv, rate);
			
				for (int i = 0; i < 1024; i++)
				{
					for (int j = 0; j < 28; j++)
					{
						for (int k = 0; k < 28; k++)
						{
							h_fcl1_weights[i*28*28 + j*28 + k] = image_layer_weights[i][j][k];
							h_fcl1_weight_deriv[i*28*28 + j*28 + k] = image_weight_deriv[i][j][k];
						}
					}
				}
				cudaMemcpy(d_fcl1_weights, h_fcl1_weights, 28*28*1024*sizeof(float), cudaMemcpyHostToDevice);
				cudaMemcpy(d_fcl1_weight_deriv, h_fcl1_weight_deriv, 28*28*1024*sizeof(float), cudaMemcpyHostToDevice);
				cudaMemcpy(d_fcl1_bias, fcl_1_bias, 1024*sizeof(float), cudaMemcpyHostToDevice);
				cudaMemcpy(d_fcl1_bias_deriv, fcl_1_bias_deriv, 1024*sizeof(float), cudaMemcpyHostToDevice);

				int blockSize = 1024;
				int gridSize = (int)ceil((float)(28*28*1024)/blockSize);
				cuda_fcl1_update<<<gridSize, blockSize>>>(d_fcl1_weights, d_fcl1_weight_deriv, d_fcl1_bias, d_fcl1_bias_deriv, rate);
				
				cudaMemcpy(h_fcl1_weights, d_fcl1_weights, 28*28*1024*sizeof(float), cudaMemcpyDeviceToHost);
				cudaMemcpy(h_fcl1_weight_deriv, d_fcl1_weight_deriv, 28*28*1024*sizeof(float), cudaMemcpyDeviceToHost);
				cudaMemcpy(fcl_1_bias, d_fcl1_bias, 1024*sizeof(float), cudaMemcpyDeviceToHost);
				cudaMemcpy(fcl_1_bias_deriv, d_fcl1_bias_deriv, 1024*sizeof(float), cudaMemcpyDeviceToHost);

				for (int i = 0; i < 1024; i++)
				{
					for (int j = 0; j < 28; j++)
					{
						for (int k = 0; k < 28; k++)
						{
							image_layer_weights[i][j][k] = h_fcl1_weights[i*28*28 + j*28 + k];
							image_weight_deriv[i][j][k] = h_fcl1_weight_deriv[i*28*28 + j*28 + k];
						}
					}
				}

				// FCL 2 Update
				// fcl_2_update(connected_layer_weights, connected_weight_deriv, fcl_2_bias, fcl_2_bias_deriv, rate);

				for (int i = 0; i < 10; i++)
				{
					for (int j = 0; j < 1024; j++)
					{
						h_fcl2_weights[i*1024 + j] = connected_layer_weights[i][j];
						h_fcl2_weight_deriv[i*1024 + j] = connected_weight_deriv[i][j];
					}
				}
				cudaMemcpy(d_fcl2_weights, h_fcl2_weights, 10*1024*sizeof(float), cudaMemcpyHostToDevice);
				cudaMemcpy(d_fcl2_weight_deriv, h_fcl2_weight_deriv, 10*1024*sizeof(float), cudaMemcpyHostToDevice);
				cudaMemcpy(d_fcl2_bias, fcl_2_bias, 10*sizeof(float), cudaMemcpyHostToDevice);
				cudaMemcpy(d_fcl2_bias_deriv, fcl_2_bias_deriv, 10*sizeof(float), cudaMemcpyHostToDevice);

				blockSize = 1024;
				gridSize = (int)ceil((float)(10*1024)/blockSize);
				cuda_fcl2_update<<<gridSize, blockSize>>>(d_fcl2_weights, d_fcl2_weight_deriv, d_fcl2_bias, d_fcl2_bias_deriv, rate);

				cudaMemcpy(h_fcl2_weights, d_fcl2_weights, 10*1024*sizeof(float), cudaMemcpyDeviceToHost);
				cudaMemcpy(h_fcl2_weight_deriv, d_fcl2_weight_deriv, 10*1024*sizeof(float), cudaMemcpyDeviceToHost);
				cudaMemcpy(fcl_2_bias, d_fcl2_bias, 10*sizeof(float), cudaMemcpyDeviceToHost);
				cudaMemcpy(d_fcl2_bias_deriv, d_fcl2_bias_deriv, 10*sizeof(float), cudaMemcpyDeviceToHost);

				for (int i = 0; i < 10; i++)
				{
					for (int j = 0; j < 1024; j++)
					{
						connected_layer_weights[i][j] = h_fcl2_weights[i*1024 + j];
						connected_weight_deriv[i][j] = h_fcl2_weight_deriv[i*1024 + j];
					}
				}
			}
			if (r % 5 == 0)
			{
				float round_time = (std::clock() - start) / (double)CLOCKS_PER_SEC;
				printf("Epoch: %d, Round: %d, Accuracy: %f, Time: %f sec (%f min)\n", epoch, r, right / (right + wrong), round_time, round_time / 60);
			}
		}
	}

	/* FORWARD */
	/*
	int forward_right = 0;
	int forward_wrong = 0;
	for (int image = 0; image < number_of_images; image++)
	{
		int label = labels[image];

		fcl_1 = first_dense_layer(images[image], image_layer_weights, fcl_1_bias);
		fcl_2 = second_dense_layer(fcl_1, connected_layer_weights, fcl_2_bias);
		sm_layer = soft_max(fcl_2);

		int index = std::distance(sm_layer, std::max_element(sm_layer, sm_layer + FCL2_DIM));
		if (index == label) forward_right++;
		else forward_wrong++;

		printf("Ratio: %f\n", (float)forward_right / float(forward_right + forward_wrong));
	}
	*/

	printf("Got through with no issues.\n");
}
