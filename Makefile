all: k-nn

k-nn: k-nn.cpp k-nn.h
	g++ -pthread k-nn.cpp -g -o k-nn -Wextra -Wall -std=c++11

clean:
	rm -f k-nn

cleandat:
	rm -f *.dat
