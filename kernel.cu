
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<cstdlib>
#include <stdio.h>
#include <vector>    // std::vector
#include <numeric>   // std::iota
#include <algorithm> // std::sort, std::nth_element, std::partial_sort
#include <cmath>
#include <random>
#include <curand_kernel.h>
#include <cfloat>

#define B 64
#define VOCAB 128000
#define Temperature 0.5
#define TopK 50
#define NUM_BLOCKS 2000


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

__device__  float values_topK[NUM_BLOCKS][B][TopK];
__device__  int indices_topK[NUM_BLOCKS][B][TopK];
__device__ int block_counter = 0;

__global__ void topK_sampling_gpu(float* input, int* output) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int offset = bid * (VOCAB/NUM_BLOCKS);

	float last_maximum = 100000000;
	int idx = 0;
	for (int i = 0;i < TopK && idx<TopK;i++) {
		float maximum= -100000000;
		int original_idx = idx;
		for (int j = 0;j < (VOCAB / NUM_BLOCKS);j++) {
			if (input[offset + tid * VOCAB + j] >= maximum && input[offset + tid * VOCAB + j] < last_maximum) {
		
				if (input[offset + tid * VOCAB + j] == maximum) {
					
					values_topK[bid][tid][original_idx] = maximum;
					indices_topK[bid][tid][original_idx] = offset + j;
					original_idx += 1;
				}
				else {
					maximum = input[offset + tid * VOCAB + j];
					original_idx = idx;
					values_topK[bid][tid][original_idx] = maximum;
					indices_topK[bid][tid][original_idx] = offset + j;
					original_idx += 1;

				}
			}
		}
			
		idx = original_idx;
		last_maximum = maximum;
	}

	__threadfence();
	int arrived = -1;
	__shared__ int last;

	if (threadIdx.x == 0) {
		arrived = atomicAdd(&block_counter, 1);
		last = (arrived == NUM_BLOCKS - 1);
	}

	__syncthreads();

	__shared__  float final_values_topK[B][TopK];
	__shared__  int final_indices_topK[B][TopK];
	last_maximum = 100000000;
	if (last) {

		idx = 0;
		for (int i = 0;i < TopK && idx < TopK;i++) {
			float maximum = -100000000;
			int original_idx = idx;
			for (int j = 0;j < NUM_BLOCKS;j++) {

				for (int k = 0;k < TopK;k++) {
					if (values_topK[j][tid][k] >= maximum && values_topK[j][tid][k] < last_maximum) {

						if (values_topK[j][tid][k] == maximum) {
							final_values_topK[tid][original_idx] = maximum;
							final_indices_topK[tid][original_idx] = indices_topK[j][tid][k];
							original_idx += 1;
						}
						else {
							original_idx = idx;
							maximum = values_topK[j][tid][k];
							final_values_topK[tid][original_idx] = maximum;
							final_indices_topK[tid][original_idx] = indices_topK[j][tid][k];
							original_idx += 1;
						}
						
					}
				}
			}

			idx = original_idx;
			last_maximum = maximum;

		}


		float maximum = final_values_topK[tid][0];
		float denominator = 0.0;
		for (int i = 0;i < TopK;i++) {
			final_values_topK[tid][i] = expf(final_values_topK[tid][i]/ Temperature - maximum);
			denominator += final_values_topK[tid][i];
		}

		float cdf = 0.0;
		curandState state;
		curand_init(0, tid, 0, &state);
		float random_sample = curand_uniform(&state);

		for (int i = 0;i < TopK;i++) {

			cdf += final_values_topK[tid][i] / denominator;

			if (random_sample < cdf) {
				output[tid] = final_indices_topK[tid][i];
				break;
			}
		}


	}
	
}

int main() {



	float* h_input = new float[B * VOCAB];
	int* h_output_cpu = new int[B];
	int* h_output = new int[B];

	float* d_input;
	int* d_output;

	cudaMalloc((void**)&d_input, B * VOCAB * sizeof(float));
	cudaMalloc((void**)&d_output, B * sizeof(int));
	

	std::mt19937 gen(123);
	std::normal_distribution<float> d(0.0f, 1.0f);

	for (int i = 0; i < B * VOCAB;i++) {
		h_input[i] = d(gen);
	}


	topk_sampling_cpu_golden(h_input, h_output_cpu);

	cudaMemcpy(d_input, h_input, B * VOCAB * sizeof(float), cudaMemcpyHostToDevice);

	int block_size = B;
	int zero = 0;
	cudaMemcpyToSymbol(block_counter, &zero, sizeof(int));
	topK_sampling_gpu << <NUM_BLOCKS, block_size >> > (d_input, d_output);

	cudaMemcpy(h_output, d_output,B * sizeof(int), cudaMemcpyDeviceToHost);



	for (int i = 0;i < B;i++) {

		printf("The Batch %d index is %d, %d\n", i, h_output_cpu[i], h_output[i]);
	}



	delete[] h_input;
	delete[] h_output_cpu;
	delete[] h_output;

	cudaFree(d_input);
	cudaFree(d_output);

}














