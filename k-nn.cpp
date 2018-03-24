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

#include "k-nn.h"

using namespace std;


/* --------------------
	Global Variables 
   -------------------- */


// Variable for brute-force algorithm
float min_found = 30000;

Node *global_root = NULL; // the top node
BestSoFar **all_results; // saves the k-nearest neighbors for each query
uint64_t num_dimensions, num_neighbors;
int point_id = 0;
int total_nodes_added = 0;
int total_points_in_tree = 0;

vector<thread *> running_threads;
static vector<ThreadJob *> job_queue;

// Mutexes/Condition Variables
mutex mtx_id;
mutex mtx_queue;
condition_variable cond_queue;


/* --------------------
	Misc. Functions 
   -------------------- */


// Returns a random 64-bit number from /dev/urandom
uint64_t generateResultID() {
	uint64_t random;
	ifstream urandom("/dev/urandom", ios::in|ios::binary);
	if (urandom) {
		urandom.read(reinterpret_cast<char*>(&random), sizeof(random));
	}
	urandom.close();
	return random;
}

// Gets the current wall time
double get_wall_time() {
	struct timeval t;
	if (gettimeofday(&t, NULL)) {
		return 0;
	}
	return (double)t.tv_sec + (double)t.tv_usec * .000001;
}

// Returns a unique ID for each new node, which is incremented safely
int getPointID() {
	mtx_id.lock();
	int number = point_id++;
	mtx_id.unlock();
	return number;
}

// Brute-force algorithm for determining the closest distance
void navigate(Node *r, Point q) {
	if (r->left != NULL) navigate(r->left, q);
	if (min_found > getDistanceBetweenPoints(q, r->point)) {
		min_found = getDistanceBetweenPoints(q, r->point);
	}
	if (r->right != NULL) navigate(r->right, q);
}


/* --------------------
	Insertion Functions 
   -------------------- */


// Adds new insertion jobs to the queue safely, notifies two threads to
// start a new job
void addToQueues(ThreadJob *t1, ThreadJob *t2, bool insertedNode) {
	mtx_queue.lock();

	// We are either adding 0, 1, or 2 new jobs to the queue, so we need to
	// know how many threads to notify
	int threads_to_notify = 0;
	
	if (insertedNode) total_nodes_added++;
	if (t1 != NULL) {
		job_queue.push_back(t1);
		threads_to_notify++;
	}
	if (t2 != NULL) {
		job_queue.push_back(t2);
		threads_to_notify++;
	}
	mtx_queue.unlock();

	// Now notify those threads
	for (int i = 0; i < threads_to_notify; i++) cond_queue.notify_one();
}

// This gets 10% of the points and returns them via sample_points
void getSamplePoints(vector<Point> &sample_points, vector<Point> points) {
	unsigned int sample_size = (points.size() / 10);
	if (sample_size == 0) sample_size = 1;

	for (unsigned int i = 0; i < sample_size; i++) {
		if (i < points.size()) sample_points.push_back(points.at(i));
	}
}

// Basic insertion algorithm, used for inserting a node into the tree
Node *insertNode(Node *root, Node *n, Node *parent, Direction direction) {
	// Set the global root to be n if it's not already set
	if (global_root == NULL) global_root = n; 

	// If out root is n, then just return the root
	if (root == n) return root;

	// If the root is NULL, we set this node as n
	if (root == NULL) {
		root = n;
		root->parent = parent;
		if (direction == Direction::left) parent->left = n; // root is parent's left child
		else parent->right = n; // root is parent's right child
		root->dimension = (parent->dimension + 1) % num_dimensions;
		return root;
	}
	
	// The root is not NULL, so keep moving down the tree
	int dim = root->dimension;
	if (root->point.coordinates[dim] >= n->point.coordinates[dim]) {
		direction = Direction::left;
		return insertNode(root->left, n, root, direction);
	}
	else {
		direction = Direction::right;
		return insertNode(root->right, n, root, direction);
	}
}

// Given a vector of points, let's add them add to the tree. We do this by 
// adding the median, and moving the left/right vectors of points into the job
// queue so some thread can do work on it
void addVectorToTree(vector<float> *points, int dimension, Node *root) {
	// We have just one point in the vector, so we can just insert this Node and be done
	if (points[dimension].size() == 1) {
		Point p;
		p.dimensions = num_dimensions;
		float *coords = new float[num_dimensions];
		for (unsigned int i = 0; i < num_dimensions; i++) {
			coords[i] = points[i].at(0);
			points[i].erase(points[i].begin() + 0);
		}
		p.coordinates = coords;
		p.id = getPointID();
		Node *node = new Node(p);
		insertNode(root, node, NULL, Direction::left);
		addToQueues(NULL, NULL, true);
		return;
	}
	else if (points[dimension].size() == 0) {
		addToQueues(NULL, NULL, false);
		return;
	}

	// Now we're assuming that "points" has more than 1 point in it, so
	// we're going to get the median point at the correct dimension
	vector<float> dimensional_points = points[dimension];
	sort(dimensional_points.begin(), dimensional_points.end());
	float split_value = dimensional_points.at(dimensional_points.size() / 2);
	int index = -1;
	for (unsigned int i = 0; i < points[dimension].size(); i++) {
		if (points[dimension].at(i) == split_value) {
			index = i;
			break;
		}
	}
	// Now remove that point and add all dimensions to "coords"
	float *coords = new float[num_dimensions];
	for (unsigned int i = 0; i < num_dimensions; i++) {
		coords[i] = points[i].at(index);
		points[i].erase(points[i].begin() + index);
	}

	// Create a new Node/point that we'll insert into the tree
	Point p;
	p.dimensions = num_dimensions;
	p.coordinates = coords;
	p.id = getPointID();
	Node *node = new Node(p);
	if (root == NULL) root = node;
	insertNode(root, node, NULL, Direction::left);

	// Now we need to create two collections of points, ones that will insert
	// to the left of the median, and ones that will insert to the right
	vector<float> *left_branch = new vector<float>[num_dimensions]; 
	vector<float> *right_branch = new vector<float>[num_dimensions];

	// Add the remaining points into either the left or right arrays
	while (points[dimension].size()) {
		float value = points[dimension].at(points[dimension].size() - 1);
		if (value < split_value) {
			// Put it in the left array
			for (unsigned int i = 0; i < num_dimensions; i++) {
				float coord = points[i].at(points[i].size() - 1);
				points[i].pop_back();
				left_branch[i].push_back(coord);
			}
		}
		else {
			// Put it in the right array
			for (unsigned int i = 0; i < num_dimensions; i++) {
				float coord = points[i].at(points[i].size() - 1);
				points[i].pop_back();
				right_branch[i].push_back(coord);
			}
		}
	}

	// Create two new jobs to be done by the threads, one for the left and right arrays
	ThreadJob *t1 = new ThreadJob(left_branch, (dimension + 1) % num_dimensions, root);
	ThreadJob *t2 = new ThreadJob(right_branch, (dimension + 1) % num_dimensions, root);
	addToQueues(t1, t2, true);
	return;
}

// This is passed to the thread, so the thread will continue to pull jobs
// from the job queue until we're inserted every node
void check_for_work(int thread_id) {
	thread_id = thread_id; // This variable is for debugging, this just removes the warning

	while (true) {
		unique_lock<mutex> lck(mtx_queue);

		while (!job_queue.size()) {
			if (total_points_in_tree <= total_nodes_added) {
				lck.unlock();
				return;
			}
			// This is used because a thread may get stuck waiting on the condition 
			// variable when it should be exiting. We wait 100 milliseconds then recheck 
			// the condition.
			cond_queue.wait_for(lck, chrono::milliseconds(100));
		}
		
		// We have the lock and the queue isn't empty, so pop a job and perform it
		ThreadJob *job = job_queue.at(job_queue.size() - 1);
		job_queue.pop_back();

		lck.unlock();
		addVectorToTree(job->points, job->dimension, job->node);
		delete job;
	}
}


/* --------------------
	Query Search Functions 
   -------------------- */


// Determines the distance between any two points
float getDistanceBetweenPoints(Point a, Point b) {
	float distance_squared = 0;
	for (int i = 0; i < a.dimensions; i++) {
		float dim_squared = a.coordinates[i] - b.coordinates[i];
		distance_squared += pow(dim_squared, 2);
	}
	return sqrt(distance_squared);
}

// This checks to see if the node "possible" with distance-from-query "poss_distance" is better than any of our neighbors
// If it is, then we replace it in the BestSoFar struct
BestSoFar *checkForBetter(BestSoFar *best, Node *possible, float poss_distance) {
	// Checks to see if this point is already in the list
	for (unsigned int i = 0; i < best->distances.size(); i++) {
		if (best->distances[i] == poss_distance) {
			return best;
		}
		if (best->nodes[i]->point.id == possible->point.id) {
			return best;
		}
	}
	// If we don't have k neighbors yet, just add it.
	if (best->distances.size() < num_neighbors) {
		best->distances.push_back(poss_distance);
		best->nodes.push_back(possible);
		return best;
	}
	// Replace the maximum element in the arrays (floats and Nodes arrays)
	int index = distance(best->distances.begin(), max_element(best->distances.begin(), best->distances.end()));
	if (best->distances.at(index) > poss_distance) {
		best->distances.erase(best->distances.begin() + index);
		best->nodes.erase(best->nodes.begin() + index);
		best->distances.push_back(poss_distance);
		best->nodes.push_back(possible);
	}
	return best;
}

// This will move to the bottom-left of the current branch, and check to see if the point is better than our neighbors
BestSoFar *getBottomValue(Node *root, Point query, BestSoFar *possible_best) {
	int dim = root->dimension;
	float root_value = root->point.coordinates[dim];
	float query_value = query.coordinates[dim];
	
	float diff = getDistanceBetweenPoints(query, root->point);
	possible_best = checkForBetter(possible_best, root, diff);
	possible_best->current = root;

	// This is a similar algorithm to the insertNode function.
	if (root_value >= query_value) {
		if (root->left == NULL) {
			if (root->right == NULL) {
				return possible_best;
			}
			else {
				return getBottomValue(root->right, query, possible_best);
			}
		}
		else {
			return getBottomValue(root->left, query, possible_best);
		}
	}
	else {
		if (root->right == NULL) {
			if (root->left == NULL) {
				return possible_best;
			}
			else {
				return getBottomValue(root->left, query, possible_best);
			}
		}
		else {
			return getBottomValue(root->right, query, possible_best);
		}
	}
}

// This is for finding the nearest neighbors
BestSoFar *getNearest(Node *root, Point query, Node *stop, BestSoFar *possible_best) {
	// Get the node that's bottom left, and work up from it
	BestSoFar *bottom_best = getBottomValue(root, query, possible_best);
	Node *move = bottom_best->current;

	// Move up until it passes the root node
	while (move != NULL) { 

		if (stop != NULL) {
			// This means we've returned to where we called the function from
			if (move->point.id == stop->point.id) break;
		}

		int dim = move->dimension;
		float distance_to_split = abs(move->point.coordinates[dim] - query.coordinates[dim]);

		// If the distance to the split dimension is closer than any of our current nearest neighbors, check that branch
		if (distance_to_split < *max_element(possible_best->distances.begin(), possible_best->distances.end())){
			// Normally we'd move left if this is true, but we already checked that branch, so check the branch to the right
			if (move->point.coordinates[dim] >= query.coordinates[dim]) {
				if (move->right != NULL) {
					possible_best = getNearest(move->right, query, move, possible_best);
				}
			}
			// Check the left branch
			else {
				if (move->left != NULL) {
					possible_best = getNearest(move->left, query, move, possible_best);
				}
			}
		}
		move = move->parent;
	}
	return possible_best;
}

// Passed to each thread; it's bascially given a list of queries it has to check
void passVectors(ComboArg *t) {
	for (unsigned int i = 0; i < t->args.size(); i++) {
		ThreadArg *arg = t->args[i];
		BestSoFar *best = getNearest(arg->root, arg->query, arg->stop, arg->possible_best);
		all_results[arg->index] = best;
	}
	delete t;
}

int main (int, char **argv) {
	vector<Point> points;
	vector<Point> queries;

	int cores = atoi(argv[1]);
	unsigned int threads = (unsigned int)(1 * cores);

	ifstream data(argv[2]);
	ifstream query(argv[3]);
	ofstream result(argv[4], ios::binary);

	// Read the first line of both files
	char data_string[9];
	char query_string[9];
	if (data.is_open()) {
		data.seekg(0);
		data.read(data_string, 8);
	}
	if (query.is_open()) {
		query.seekg(0);
		query.read(query_string, 8);
	}

	data_string[8] = '\0';
	query_string[8] = '\0';
	assert(strcmp(data_string, "TRAINING") == 0);
	assert(strcmp(query_string, "QUERY") == 0);

	// Read the ID/rows/columns from the data file and ID/entries/dimensions/num_neighbors from query file
	uint64_t data_id, rows, columns;
	uint64_t query_id, num_queries;
	data.read(reinterpret_cast<char*>(&data_id), sizeof(uint64_t));
	data.read(reinterpret_cast<char*>(&rows), sizeof(uint64_t));
	data.read(reinterpret_cast<char*>(&columns), sizeof(uint64_t));
	query.read(reinterpret_cast<char*>(&query_id), sizeof(uint64_t));
	query.read(reinterpret_cast<char*>(&num_queries), sizeof(uint64_t));
	query.read(reinterpret_cast<char*>(&num_dimensions), sizeof(uint64_t));
	query.read(reinterpret_cast<char*>(&num_neighbors), sizeof(uint64_t));

	cout << "Points:" << rows << ", Dimensions:" << num_dimensions << ", Queries:" << num_queries << ", Neighbors:" << num_neighbors << endl;

	assert(columns == num_dimensions);
	total_points_in_tree = rows;

	double start_tree_creation = get_wall_time();
	cout << "Beginning tree construction... (" << threads << " thread" << (threads > 1 ? "s)" : ")") << endl;

	// vector<float> points_by_dimension[num_dimensions];
	vector<float> *points_by_dimension = new vector<float>[num_dimensions];
	// Add all the points into an array (by dimension) of vectors
	for (uint64_t i = 0; i < rows; i++) {
		for (uint64_t j = 0; j < columns; j++) {
			float value;
			data.read(reinterpret_cast<char*>(&value), sizeof(float));
			points_by_dimension[j].push_back(value);
		}
	}
	
	// Add all the queries into a vector of points
	for (uint64_t i = 0; i < num_queries; i++) {
		Point temp_point;
		temp_point.dimensions = num_dimensions;
		temp_point.coordinates = new float[num_dimensions];
		for (uint64_t j = 0; j < num_dimensions; j++) {
			float value;
			query.read(reinterpret_cast<char*>(&value), sizeof(float));
			temp_point.coordinates[j] = value;
		}
		queries.push_back(temp_point);
	}

	// Close the input streams
	query.close();
	data.close();


	/*  ----------------------------------
			Begin Building Tree
		---------------------------------- */


	for (unsigned int i = 0; i < threads; i++) {
		running_threads.push_back(new thread(check_for_work, i));
	}

	job_queue.push_back(new ThreadJob(points_by_dimension, 0, NULL));
	
	for (unsigned int i = 0; i < running_threads.size(); i++) {
		running_threads[i]->join();
		delete running_threads[i];
	}

	cout << "-> Total time: " << get_wall_time() - start_tree_creation << " sec" << endl;


	/*  ----------------------------------
			Execute the Queries
		---------------------------------- */
	

	double start_query_execution = get_wall_time();
	cout << "Beginning query execution... (" << threads << " thread" << (threads > 1 ? "s)" : ")") << endl;

	/*
	// Prints the nearest neighbor to the query point (using bruteforce algorithm)
	for (unsigned int i = 0; i < queries.size(); i++) {
		min_found = 30000;
		navigate(global_root, queries[i]);
		cout << "Query " << i << " is: " << min_found << endl;
	}
	*/

	all_results = new BestSoFar *[queries.size()];

	vector<BestSoFar*> guesses[threads];
	vector<ThreadArg*> args[threads];
	thread all_threads[threads];
	for (unsigned int i = 0; i < queries.size(); i++) {
		int index_of_thread = i % threads;
		guesses[index_of_thread].push_back(new BestSoFar(global_root, getDistanceBetweenPoints(global_root->point, queries[i]), NULL));
		args[index_of_thread].push_back(new ThreadArg(global_root, queries[i], NULL, guesses[index_of_thread][guesses[index_of_thread].size() - 1], (int)(i)));
	}
	for (unsigned int i = 0; i < threads; i++) {
		ComboArg *t = new ComboArg(args[i], guesses[i]);
		all_threads[i] = thread(passVectors, t);
	}
	for (unsigned int i = 0; i < threads; i++) {
		all_threads[i].join();
	}

	/*
	// Prints out distances to k nearest neighbors (using normal algorithm)
	for (unsigned int i = 0; i < num_queries; i++) {
		cout << "SQuery " << i << ": ";
		for (unsigned int j = 0; j < num_neighbors; j++) {
			cout << all_results[i]->distances[j] << " ";
		}
		cout << endl;
	}
	*/

	cout << "-> Total time: " << get_wall_time() - start_query_execution << " sec" << endl;


	/*  ----------------------------------
			Write to File
		---------------------------------- */


	if (result.is_open()) {
		char message[8] = {'R', 'E', 'S', 'U', 'L', 'T'};
		uint64_t result_id = generateResultID();
		result.write(message, sizeof(message));
		result.write(reinterpret_cast<char*>(&data_id), sizeof(uint64_t));
		result.write(reinterpret_cast<char*>(&query_id), sizeof(uint64_t));
		result.write(reinterpret_cast<char*>(&result_id), sizeof(uint64_t));
		result.write(reinterpret_cast<char*>(&num_queries), sizeof(uint64_t));
		result.write(reinterpret_cast<char*>(&num_dimensions), sizeof(uint64_t));
		result.write(reinterpret_cast<char*>(&num_neighbors), sizeof(uint64_t));

		for (unsigned int i = 0; i < num_queries; i++) {
			for (unsigned int j = 0; j < num_neighbors; j++) {
				for (unsigned int k = 0; k < num_dimensions; k++) {
					float value = (all_results[i])->nodes[j]->point.coordinates[k];
					result.write(reinterpret_cast<char*>(&value), sizeof(float));
				}
			}
		}
	}
	else {
		cout << "Can't access result file. Exiting." << endl;
	}
	result.close();
	
	// Delete all allocated memory on heap
	for (unsigned int i = 0; i < queries.size(); i++) delete []queries[i].coordinates; // delete coordinates of queries
	for (unsigned int i = 0; i < num_queries; i++) delete all_results[i]; // delete the k-nearest neighbor results
	for (unsigned int i = 0; i < threads; i++) {
		for (unsigned int j = 0; j < args[i].size(); j++) delete args[i][j]; // delete the args passed to threads
	}
	delete []all_results; // delete the array
	delete global_root; // iterate through all nodes and delete them
	return 0;
}
