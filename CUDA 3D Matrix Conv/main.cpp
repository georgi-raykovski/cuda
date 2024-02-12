#include "CpuGpuMat.h"
#include <stdlib.h>
#include <assert.h>
#include "KernelGpu.cuh"
#include <cuda_runtime_api.h>
#include <iostream>
#include <chrono>

// Максималното ускорение, което успях да постигна като изпълнявам кода в Google Colab:
// Input Size: 10, Global Memory Time: 0.233792 ms, Shared Memory Time: 0.022528 ms, Speedup: 10.3778
// Input Size: 20, Global Memory Time: 0.014368 ms, Shared Memory Time: 0.010784 ms, Speedup: 1.33234
// Input Size: 50, Global Memory Time: 0.013344 ms, Shared Memory Time: 0.01104 ms, Speedup: 1.2087
// Input Size: 100, Global Memory Time: 0.0128 ms, Shared Memory Time: 0.011392 ms, Speedup: 1.1236
// Input Size: 200, Global Memory Time: 0.013888 ms, Shared Memory Time: 0.011296 ms, Speedup: 1.22946
// Input Size: 400, Global Memory Time: 0.012352 ms, Shared Memory Time: 0.01024 ms, Speedup: 1.20625

// Command used to compile -> nvcc main.cpp KernelGpu.cu -o conv3D

// Function to measure the execution time of a kernel
float measureKernelTime(void (*kernel)(struct CpuGpuMat *, struct CpuGpuMat *, struct CpuGpuMat *),
												struct CpuGpuMat *image, struct CpuGpuMat *mask, struct CpuGpuMat *result)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	kernel(image, mask, result);
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return milliseconds;
}

int main()
{
	struct CpuGpuMat Mat1;
	struct CpuGpuMat Mat2;
	struct CpuGpuMat Mat3;
	int maskSize = 3;

	Mat1.Rows = 4;
	Mat1.Cols = 4;
	Mat1.Depth = 3;

	Mat2.Rows = maskSize;
	Mat2.Cols = maskSize;
	Mat2.Depth = 3;

	Mat3.Rows = Mat1.Rows - maskSize + 1;
	Mat3.Cols = Mat1.Cols - maskSize + 1;
	Mat3.Depth = 1;

	Mat1.Size = Mat1.Rows * Mat1.Cols * Mat1.Depth;
	Mat2.Size = Mat2.Rows * Mat2.Cols * Mat2.Depth;
	Mat3.Size = Mat3.Rows * Mat3.Cols * Mat3.Depth;

	// cpu and gpu memory allocation
	Mat1.cpuP = (void *)malloc(Mat1.Size * sizeof(float));
	Mat2.cpuP = new float[Mat2.Size]{0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370,	// mean filter
																	 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370,	// mean filter
																	 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370, 0.0370}; // mean filter
	Mat3.cpuP = (void *)malloc(Mat3.Size * sizeof(float));

	cudaError_t result1 = cudaMalloc(&Mat1.gpuP, Mat1.Size * sizeof(float));
	cudaError_t result2 = cudaMalloc(&Mat2.gpuP, Mat2.Size * sizeof(float));
	cudaError_t result3 = cudaMalloc(&Mat3.gpuP, Mat3.Size * sizeof(float));
	assert(result1 == cudaSuccess || result2 == cudaSuccess || result3 == cudaSuccess);

	// set values to cpu memory
	float *cpuFloatP = (float *)Mat1.cpuP;
	for (int i = 0; i < Mat1.Size; i++)
		cpuFloatP[i] = (float)i;

	//	Host => ram
	//	Device => graphics memory

	// Host -> Device
	result1 = cudaMemcpy(Mat1.gpuP, Mat1.cpuP, Mat1.Size * sizeof(float), cudaMemcpyHostToDevice);
	result2 = cudaMemcpy(Mat2.gpuP, Mat2.cpuP, Mat2.Size * sizeof(float), cudaMemcpyHostToDevice);
	result3 = cudaMemcpy(Mat3.gpuP, Mat3.cpuP, Mat3.Size * sizeof(float), cudaMemcpyHostToDevice);
	assert(result1 == cudaSuccess || result2 == cudaSuccess || result3 == cudaSuccess);

	// Perform benchmarking for different input sizes
	int inputSizes[] = {10, 20, 50, 100, 200, 400};
	for (int i = 0; i < sizeof(inputSizes) / sizeof(inputSizes[0]); ++i)
	{
		Mat1.Rows = Mat1.Cols = Mat1.Depth = inputSizes[i];

		auto globalMemoryTime = measureKernelTime(gpuMatrixConvulation3D, &Mat1, &Mat2, &Mat3);
		auto sharedMemoryTime = measureKernelTime(gpuMatrixConvulation3DShared, &Mat1, &Mat2, &Mat3);

		float speedup = globalMemoryTime / sharedMemoryTime;

		std::cout << "Input Size: " << inputSizes[i] << ", Global Memory Time: " << globalMemoryTime << " ms, "
							<< "Shared Memory Time: " << sharedMemoryTime << " ms, "
							<< "Speedup: " << speedup << std::endl;
	}

	// cpu and gpu memory free
	result1 = cudaFree(Mat1.gpuP);
	result2 = cudaFree(Mat2.gpuP);
	result3 = cudaFree(Mat3.gpuP);
	assert(result1 == cudaSuccess || result2 == cudaSuccess || result3 == cudaSuccess);

	free(Mat1.cpuP);
	free(Mat2.cpuP);
	free(Mat3.cpuP);

	return 0;
}
