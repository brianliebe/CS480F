FLAGS = -g #-Wextra -Wall

all: seq cuda

seq: seq.cpp
	g++ seq.cpp -o seq -std=c++11 $(FLAGS)

cuda: cuda.cu
	nvcc cuda.cu -o cuda -std=c++11 -Wno-deprecated-gpu-targets

clean:
	rm -f prog3 seq cuda
