
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<cstdlib>
#include <stdio.h>
#include <vector>    // std::vector
#include <numeric>   // std::iota
#include <algorithm> // std::sort, std::nth_element, std::partial_sort
#include <cmath>
#include <random>

#define B 128
#define VOCAB 128000
#define Temperature 0.5
#define TopK 50


void topk_sampling_cpu_golden(float* input, int* output) {

	std::mt19937 rng(42); // seed

	//Find TopK
	int* topK_indices = new int[B * TopK];
	float* topK_values = new float[B * TopK];
	std::uniform_real_distribution<float> dist(0.0f, 1.0f);
	for (int i = 0;i < B;i++) {
		int32_t offset = VOCAB * i;

		std::vector<int> indices(VOCAB);
		std::iota(indices.begin(), indices.end(), 0);
		std::nth_element(indices.begin(), indices.begin() + TopK, indices.end(),
			[&](int a, int b) { return input[offset+a] > input[offset+b]; });

		std::sort(indices.begin(), indices.begin() + TopK,
			[&](int a, int b) { return input[offset + a] > input[offset + b]; });

		float maximum = input[offset + indices[0]] / Temperature;
		float denominator = 0.0;
		for (int k = 0;k < TopK;k++) {
			topK_indices[TopK * i + k] = indices[k];
			float exponent = std::exp(input[offset+indices[k]] / Temperature - maximum);
			denominator += exponent;
			topK_values[TopK * i + k] = exponent;
		}

		float random_number = dist(rng);
		float cdf = 0.0f;
		output[i] = topK_indices[TopK * i + TopK-1];
		for (int k = 0;k < TopK;k++) {

			cdf+= topK_values[TopK * i + k]/denominator;

			if(random_number <= cdf){

				output[i] = topK_indices[TopK * i + k];
				break;
			}
		}

	}

	delete[] topK_indices;
	delete[] topK_values;

}


int main() {



	float* input = new float[B * VOCAB];
	int* output = new int[B];
	std::mt19937 gen(123);
	std::normal_distribution<float> d(0.0f, 1.0f);

	for (int i = 0; i < B * VOCAB;i++) {
		input[i] = d(gen);
	}


	topk_sampling_cpu_golden(input, output);

	for (int i = 0;i < B;i++) {

		printf("The Batch %d index is %d\n", i, output[i]);
	}

	delete[] input;
	delete[] output;

}














