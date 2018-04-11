#include <iostream>
#include <vector>
#include <math.h>
#include <sys/time.h>
#include <random>
#include <immintrin.h>
#include <chrono>
#include <cmath>
#include <functional>

using namespace std;

const int SIZE = 16'000'000;

double get_wall_time() {
	struct timeval t;
	if (gettimeofday(&t, NULL)) {
		return 0;
	}
	return (double)t.tv_sec + (double)t.tv_usec * .000001;
}

double time(const function<void ()> &f) {
    auto start = chrono::system_clock::now();
    f();
    auto stop = chrono::system_clock::now();
    return chrono::duration<double>(stop - start).count();
}

int main () {
	cout << "Generating data of size " << SIZE << endl;

	// create 8 arrays (one per dimension for each point) on the heap
	alignas(32) static float *a1 = new float[SIZE];
	alignas(32) static float *a2 = new float[SIZE];
	alignas(32) static float *b1 = new float[SIZE];
	alignas(32) static float *b2 = new float[SIZE];
	alignas(32) static float *c1 = new float[SIZE];
	alignas(32) static float *c2 = new float[SIZE];
	alignas(32) static float *d1 = new float[SIZE];
	alignas(32) static float *d2 = new float[SIZE];

	default_random_engine eng;
	uniform_real_distribution<float> dist(-1, 1);

	// generate SIZE * 8 random numbers, -1 to 1
	for (int i = 0; i < SIZE; i++) {
		a1[i] = dist(eng);
		a2[i] = dist(eng);
		b1[i] = dist(eng);
		b2[i] = dist(eng);
		c1[i] = dist(eng);
		c2[i] = dist(eng);
		d1[i] = dist(eng);
		d2[i] = dist(eng);
	}

	/* SEQUENTIAL */

	double start_seq = get_wall_time();

	// store results in a float array on heap
	float *seq_results = new float[SIZE];

	auto seq = [&]() {
		for (unsigned int i = 0; i < SIZE; i++) {
			// basically, result[i] = sqrt((a2 - a1)^2 + (b2 - b1)^2 + (c2 - c1)^2 + (d2 - d1)^2)
			seq_results[i] = sqrtf(powf(a2[i] - a1[i], 2) + powf(b2[i] - b1[i], 2) + powf(c2[i] - c1[i], 2) + powf(d2[i] - d1[i], 2));
		}
	};

    cout << "Seqential: " << (SIZE/time(seq))/1000000 << " Mops/s (";
	cout << get_wall_time() - start_seq << " sec)" << endl;

	/* SIMD/AVX */

	double start_simd = get_wall_time();

	// save results to float array on heap
	alignas(32) static float *simd_results = new float[SIZE];

	auto vec = [&]() {
		for (int i = 0; i < SIZE/8; i++) {
			// load all the values from each array
			__m256 ymm_a1 = _mm256_loadu_ps(a1 + i*8);
			__m256 ymm_a2 = _mm256_loadu_ps(a2 + i*8);
			__m256 ymm_b1 = _mm256_loadu_ps(b1 + i*8);
			__m256 ymm_b2 = _mm256_loadu_ps(b2 + i*8);
			__m256 ymm_c1 = _mm256_loadu_ps(c1 + i*8);
			__m256 ymm_c2 = _mm256_loadu_ps(c2 + i*8);
			__m256 ymm_d1 = _mm256_loadu_ps(d1 + i*8);
			__m256 ymm_d2 = _mm256_loadu_ps(d2 + i*8);

			// subtract x2 - x1 for a through d
			__m256 ymm_a = _mm256_sub_ps(ymm_a2, ymm_a1);
			__m256 ymm_b = _mm256_sub_ps(ymm_b2, ymm_b1);
			__m256 ymm_c = _mm256_sub_ps(ymm_c2, ymm_c1);
			__m256 ymm_d = _mm256_sub_ps(ymm_d2, ymm_d1);

			// save the sqrt of the sum of all those subtractions we just did
			__m256 ymm_l = _mm256_sqrt_ps(_mm256_mul_ps(ymm_a, ymm_a) + _mm256_mul_ps(ymm_b, ymm_b) + _mm256_mul_ps(ymm_c, ymm_c) + _mm256_mul_ps(ymm_d, ymm_d));

			// store that value as a result
			_mm256_storeu_ps(simd_results + i*8, ymm_l);
		}
	};

	cout << "Vector/AVX: " << (SIZE/time(vec))/1000000 << " Mops/s (";
	cout << get_wall_time() - start_simd << " sec)" << endl;

	cout << "Errors found: ";
	// check for any errors
	int errors = 0;
	for (int i = 0; i < SIZE; i++) {
		if (seq_results[i] != simd_results[i]) {
			errors++;
		}
	}
	if (errors == 0) cout << "None" << endl;
	else cout << errors << endl;

	// delete the heaped data
	(delete [] a1, delete [] a2, delete [] b1, delete [] b2, delete [] c1, delete [] c2, delete [] d1, delete [] d2);
	(delete [] simd_results, delete[] seq_results);
}
