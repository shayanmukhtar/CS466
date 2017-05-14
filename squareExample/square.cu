
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 *
 * See cuda.h for error code descriptions.
 */
#define CHECK_CUDA_RESULT(N) {											\
	CUresult result = N;												\
	if (result != 0) {													\
		printf("CUDA call on line %d returned error %d\n", __LINE__,	\
			result);													\
		exit(1);														\
	} }

/*Thread - take in a number and square it*/
__global__ void square(float * d_in, float * d_out)
{
	/*threadIdx is actually a C struct with three members
	 * x, y, z - we only need x right now*/
	int threadId = threadIdx.x;
	float data = d_in[threadId];
	d_out[threadId] = data * data;
}

int main(int argc, char **argv)
{
	const int ARRAY_SIZE = 64;
	const int ARRAY_BYTES = 64 * sizeof(float);

	/*Allocate CPU memory*/
	float * h_in = (float *) malloc(ARRAY_BYTES);
	float * h_out = (float *) malloc(ARRAY_BYTES);

	/*Declare GPU pointers*/
	float * d_in;
	float * d_out;

	for (int index = 0; index < ARRAY_SIZE; index++)
	{
		/*Fill in host array*/
		h_in[index] = float(index);
	}

	/*Allocate memory on the GPU*/
	cudaMalloc((void **) &d_in, ARRAY_BYTES);
	cudaMalloc((void **) &d_out, ARRAY_BYTES);

	/*Now memory is allocated and filled on CPU side, and allocated on GPU side
	 * Next step is to copy the input array from the CPU to the GPU*/

	cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);	//Host is CPU, device is GPU

	/*Launch the GPU kernel
	 * 1 block
	 * ARRAY_SIZE threads in the block*/
	square<<<1, ARRAY_SIZE>>>(d_in, d_out);

	/*the kernel call is blocking?
	 * Anyways copy from d_out to h_out*/
	cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	/*On CPU, print results to ensure correctness*/

	for (int index = 0; index < ARRAY_SIZE; index++)
	{
		printf("Num: %f \t Num Squared: %f\n", h_in[index], h_out[index]);

	}

	/*Never forget to free the memory when you are done*/
	free(h_out);
	free(h_in);
	cudaFree(d_out);
	cudaFree(d_in);
	return 0;
}
