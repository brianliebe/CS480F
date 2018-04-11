#ifndef KNN_H
#define KNN_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cassert>
#include <sys/time.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <iterator>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>

using namespace std;


/* --------------------
	Structs & Enums 
   -------------------- */


// Used for saying which direction the node moved in the tree
// (left/right) so we can assign the parent easier
enum Direction { right, left };

// The point within each node in the k-d tree, which stores
// all the info from the input file
typedef struct Point Point;
struct Point {
	int dimensions;
	float *coordinates;
	int id;
};

// A basic node like in a BST, with left/right/parent, as well
//  as a Point and the dimension that's split at this node
typedef struct Node Node;
struct Node {
	Node *right;
	Node *left;
	Node *parent;
	int dimension;
	Point point;
	Node (Point p) {
		right = NULL;
		left = NULL;
		parent = NULL;
		dimension = 0;
		point = p;
	}
	~Node() {
		if (right != NULL) delete right;
		if (left != NULL) delete left;
		if (point.coordinates != NULL) delete [] point.coordinates;
		left = right = NULL;
	}
};

// Saves the current closest neighbors as floats and as points,
// so we don't have to recalculate distances for each nearest neighbor
typedef struct BestSoFar BestSoFar;
struct BestSoFar {
	Node *nearest;
	Node *current;
	vector<float> distances;
	vector<Node *> nodes;
	BestSoFar(Node *n1, float d, Node *n2) {
		nodes.push_back(n1);
		current = n2;
		distances.push_back(d);
	}
};

// This is passed to the query search algorithm 
typedef struct ThreadArg ThreadArg;
struct ThreadArg {
	Node *root;
	Point query;
	Node *stop;
	BestSoFar *possible_best;
	int index;
	ThreadArg(Node *r, Point p, Node *s, BestSoFar *b, int ind) {
		root = r;
		query = p;
		stop = s;
		possible_best = b;
		index = ind;
	}
};

// A vector of ThreadArg and BestSoFar, to aid the parallelization 
// of the query searching
typedef struct ComboArg ComboArg;
struct ComboArg {
	vector<ThreadArg*> args;
	vector<BestSoFar*> guesses;
	ComboArg(vector<ThreadArg*> a, vector<BestSoFar*> b) {
		args = a;
		guesses = b;
	}
};

// Used for insertion, this is passed to a thread 
typedef struct ThreadJob ThreadJob;
struct ThreadJob {
	vector<float> *points;
	int dimension;
	Node *node;
	int thread_id;
	ThreadJob(vector<float> *p, int d, Node *n) {
		points = p;
		dimension = d;
		node = n;
	}
	~ThreadJob() {
		delete [] points;
	}
};

// Misc functions
uint64_t generateResultID();
double get_wall_time();
int getPointID();
void navigate(Node *r, Point q);

// Insertion functions
void addToQueues(ThreadJob *, ThreadJob *, bool);
void getSamplePoints(vector<Point> &, vector<Point>);
Node *insertNode(Node *, Node *, Node *, Direction);
void addVectorToTree(vector<float>, int, Node *);
void check_for_work(int);

// Query functions
float getDistanceBetweenPoints(Point, Point);
BestSoFar *checkForBetter(BestSoFar *, Node *, float);
BestSoFar *getBottomValue(Node *, Point, BestSoFar *);
BestSoFar *getNearest(Node *, Point, Node *, BestSoFar *);
void passVectors(ComboArg *);


#endif
