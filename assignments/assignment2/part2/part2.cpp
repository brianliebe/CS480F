#include <iostream>
#include <random>
#include <vector>
#include <math.h>
#include <algorithm>
#include <iomanip>
#include <sys/ioctl.h>
#include <unistd.h>
#include <sys/time.h>
#include <omp.h>
#include <cassert>
#include <fstream>

using namespace std;

int dimensions, num_points, max_dimension;
default_random_engine eng;
uniform_real_distribution<float> uniform_dist(0, 1);
normal_distribution<float> normal_dist(0, 1);

typedef struct Point {
    vector<float> coordinates;
} Point;

double get_wall_time() {
	struct timeval t;
	if (gettimeofday(&t, NULL)) {
		return 0;
	}
	return (double)t.tv_sec + (double)t.tv_usec * .000001;
}

float distance_to_center(Point *p) {
    float sum = 0;
    for (int i = 0; i < dimensions; i++) {
        sum += powf(p->coordinates[i], 2);
    }
    return sqrtf(sum);
}

bool sort_points(Point *a, Point *b) {
    float dist_a = distance_to_center(a);
    float dist_b = distance_to_center(b);
    return dist_a > dist_b;
}

int main(int argc, char **argv) {
    // cout will only display 2 numbers after decimal
    cout << fixed;
    cout << setprecision(2);


    /* -- Read arguments -- */


    if (argc != 4 && argc != 5) {
        cout << "USAGE:  part2 <min dimension> <max dimension> <num points> <opt: 0/1>" << endl;
        cout << "\tRead README for more information" << endl;
        return 0;
    }

    dimensions = atoi(argv[1]);
    max_dimension = atoi(argv[2]);
    num_points = atoi(argv[3]);
    bool print_points = false;
    if (argc == 5) print_points = atoi(argv[4]);

    // output stream
    ofstream output("output.csv");
    for (int i = 0; i < dimensions - 1; i++) {
        output << "Dim" << i << ",";
    }
    output << "Dim" << dimensions - 1 << endl;


    /* -- Sequential -- */ 


    double start_seq = get_wall_time();

    vector<Point *> points;
    float interval_size = 0.01;

	for (int i = 0; i < num_points; i++) {
        vector<float> coordinates;
        Point *p = new Point();
        
        // add Normally distributed N-dimensions (-1 to 1)
        for (int j = 0; j < dimensions; j++) coordinates.push_back(normal_dist(eng));

        // define "r", which will move all coordinates to the surface of the n-sphere with Xi / r
        float r = 0;
        for (int j = 0; j < dimensions; j++) r += powf(coordinates[j], 2);
        r = sqrt(r);
        for (int j = 0; j < dimensions; j++) coordinates[j] = coordinates[j] / r;

        // define "u", a uniformally distributed float (0 to 1)
        // shift all values with Xi * r to distribute correctly within the circle
        float u = abs(uniform_dist(eng));
        r = pow(u, 1.0/dimensions);
        for (int j = 0; j < dimensions; j++) {
            coordinates[j] = coordinates[j] * r;
        }
        // add point to points, no need to check for outliers 
        p->coordinates = coordinates;
        points.push_back(p);
    }

    int totals[100]; // saves the totals for each interval from the surface
    for (int i = 0; i < 100; i++) totals[i] = 0;

    // get the distance for each point, and add it to "totals"
    for (unsigned int i = 0; i < points.size(); i++) {
        Point *p = points.at(i);
        float distance = distance_to_center(p);
        // index into "totals" i.e. 0.67890 -> 67 meaning index 67 in totals
        int location = distance * 100;
        totals[location]++;
    }

    float interval = 0.0;
    for (int i = 0; i < 100; i++) {
        // print out data for this interval, and increase the interval
        cout << interval << "-" << interval + interval_size << ": ";
        cout << totals[i] << endl;

        interval += 0.01;
        interval = floor(interval * 100 + 0.5) / 100;
    }
    cout << "Total time (Sequential - " << dimensions << " dimensions" << "): " << get_wall_time() - start_seq << " seconds" << endl;
    
    while (points.size()) {
        // delete all the Points on the heap 
        Point *p = points[points.size() - 1];
        points.pop_back();
        
        delete p;
        p = NULL;
    }


    /* -- Parallel OpenMP -- */



    // OpenMP lock for accessing "points" and "totals"
    omp_lock_t lock;
    omp_init_lock(&lock);

    while (dimensions <= max_dimension) {
        double start_openmp = get_wall_time();

        #pragma omp parallel for
        for (int i = 0; i < num_points; i++) {
            vector<float> coordinates;
            
            // add Normally distributed N-dimensions (-1 to 1)
            for (int j = 0; j < dimensions; j++) coordinates.push_back(normal_dist(eng));

            // define "r", which will move all coordinates to the surface of the n-sphere with Xi / r
            float r = 0;
            for (int j = 0; j < dimensions; j++) r += powf(coordinates[j], 2);
            r = sqrt(r);
            for (int j = 0; j < dimensions; j++) coordinates[j] = coordinates[j] / r;

            // define "u", a uniformally distributed float (0 to 1)
            // shift all values with Xi * r to distribute correctly within the circle
            float u = abs(uniform_dist(eng));
            r = pow(u, 1.0/dimensions);
            for (int j = 0; j < dimensions; j++) coordinates[j] = coordinates[j] * r;

            // add point to points, no need to check for outliers 
            Point *p = new Point();
            p->coordinates = coordinates;
            
            // Add to points, but acquire a lock first
            omp_set_lock(&lock);
            points.push_back(p);
            omp_unset_lock(&lock);
        }

        #pragma omp parallel for
        for (int i = 0; i < 100; i++) totals[i] = 0; // reset all to 0

        #pragma omp parallel for
        for (unsigned int i = 0; i < points.size(); i++) {
            // same as before, get an index into the totals array
            Point *p = points.at(i);
            float distance = distance_to_center(p);
            int location = distance * 100;

            // acquire the lock, then increment
            omp_set_lock(&lock);
            totals[location]++;
            omp_unset_lock(&lock);
        }

        interval = 0.0;
        for (int i = 0; i < 100; i++) {
            // print out # points at this interval
            cout << interval << "-" << interval + interval_size << ": ";
            cout << totals[i] << endl;

            // print this info to the CSV file
            if (!print_points) {
                float percentage = (float)totals[i] / (float)num_points;
                output << dimensions << "," << interval << "," << percentage * 100 << endl;
            }

            interval += 0.01;
            interval = floor(interval * 100 + 0.5) / 100;
        }

        cout << "Total time (OpenMP - " << dimensions << " dimensions" << "): " << get_wall_time() - start_openmp << " seconds" << endl;

        while (points.size()) {
            // delete the newly created points
            Point *p = points[points.size() - 1];
            points.pop_back();
            
            // prints the coordinates to the .CSV file, rather than the percentage/distance/dimension data
            if (print_points) {
                for (int i = 0; i < dimensions - 1; i++) {
                    output << p->coordinates[i] << ",";
                }
                output << p->coordinates[dimensions - 1] << endl;
            }

            delete p;
            p = NULL;
        }
        dimensions++;
    }
    output.close();
}
