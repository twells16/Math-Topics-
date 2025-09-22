// Name:Tanner Wells
// Vector Dot product on 1 block 
// nvcc HW8.cu -o temp
/*
 What to do:
 This code uses the CPU to compute the dot product of two vectors of length N. 
 It includes a skeleton for setting up a GPU dot product, but that part is currently empty.
 Additionally, the CPU code is somewhat convoluted, but it is structured this way to parallel 
 the GPU code you will need to write. The program will also verify whether you have correctly 
 implemented the dot product on the GPU.
 Leave the block and vector sizes as:
 Block = 1000
 N = 823
 Use folding at the block level when you do the addition reduction step.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <math.h>

// Defines
#define N 823 // Length of the vector

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
float DotCPU, DotGPU;
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float Tolerance = 0.01;

// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices();
void allocateMemory();
void innitialize();
float dotProductCPU_Simple(float*, float*, int);
__global__ void dotProductGPU(float*, float*, float*, int);
bool  check(float, float, float);
long elaspedTime(struct timeval, struct timeval);
void CleanUp();

// This check to see if an error happened in your CUDA code. It tells you what it thinks went wrong,
// and what file and line it occurred on.
void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}

// This will be the layout of the parallel space we will be using.
void setUpDevices()
{
	BlockSize.x = 1024;  // Updated to 1024 threads per block (power of two)
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = 1;      // 1 block (fixed)
	GridSize.y = 1;
	GridSize.z = 1;
}

// Allocating the memory we will be using.
void allocateMemory()
{	
	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(sizeof(float)); // Only need one float to store final dot product result
	
	// Device "GPU" Memory
	cudaMalloc(&A_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU,sizeof(float));  // Only one float on device for final result
	cudaErrorCheck(__FILE__, __LINE__);
}

// Loading values into the vectors that we will add.
void innitialize()
{
	for(int i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)(2*i);
	}
}

// Adding vectors a and b on the CPU then returns the dot product sum.
float dotProductCPU_Simple(float *a, float *b, int n)
{
	float sum = 0.0f;
    for (int i = 0; i < n; i++)
    {
        sum += a[i] * b[i];
    }
    return sum;
}

// This is the kernel. It is the function that will run on the GPU.
//This function is going to compute the dot product on the GPU
//This happens by creating a shared memory space to store partial products that will be computed in
//this function. You then write an if, else loop to make the threads compute a partial product as 
//long as it is in bounds. You then synce the threads to make sure they aren't leaving anyone behind
//You then write a for loop that will perform block level reduction, which is a parallel technique that 
//makes all the threads within a single block work together to compute a large calculation.
//You then write a an if loop to write the results to global memory.
__global__ void dotProductGPU(float *a, float *b, float *C_GPU, int n)
{
	// Shared memory to store partial products
	__shared__ float partialSum[1024];  // Updated to 1024 to match block size

	int tid = threadIdx.x;

	// Each thread computes product or 0 if out of range
	if(tid < n)
		partialSum[tid] = a[tid] * b[tid];
	else
		partialSum[tid] = 0.0f;

	__syncthreads();

	// Block level reduction: parallel sum within the block
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
		{
			partialSum[tid] += partialSum[tid + stride];
		}
		__syncthreads();
	}

	// First thread writes result to global memory
	if (tid == 0)
		C_GPU[0] = partialSum[0];
}

// Checking to see if anything went wrong in the vector addition.
bool check(float cpuAnswer, float gpuAnswer, float tolerance)
{
	double percentError;
	
	percentError = fabs((gpuAnswer - cpuAnswer) / cpuAnswer) * 100.0;
	printf("\n\n percent error = %lf\n", percentError);
	
	return (percentError < tolerance);
}

// Calculating elapsed time.
long elaspedTime(struct timeval start, struct timeval end)
{
	long startTime = start.tv_sec * 1000000 + start.tv_usec; // In microseconds.
	long endTime = end.tv_sec * 1000000 + end.tv_usec; // In microseconds

	return endTime - startTime;
}

// Cleaning up memory after we are finished.
void CleanUp()
{
	free(A_CPU); 
	free(B_CPU); 
	free(C_CPU);
	
	cudaFree(A_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C_GPU);
	cudaErrorCheck(__FILE__, __LINE__);
}

int main()
{
	timeval start, end;
	long timeCPU, timeGPU;
	
	// Setup GPU grid/block sizes
	setUpDevices();
	
	// Allocate memory on host and device
	allocateMemory();
	
	// Initialize input vectors on host
	innitialize();
	
	// Compute dot product on CPU and time it
	gettimeofday(&start, NULL);
	DotCPU = dotProductCPU_Simple(A_CPU, B_CPU, N);
	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);
	
	// Copy memory from CPU to GPU, synchronous copy to avoid race conditions
	cudaMemcpy(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpy(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);

	// Zero out C_GPU on device before kernel launch (optional but safe)
	cudaMemset(C_GPU, 0, sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Launch kernel to compute dot product on GPU
	gettimeofday(&start, NULL);
	dotProductGPU<<<GridSize, BlockSize>>>(A_GPU, B_GPU, C_GPU, N);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Copy result back from GPU to CPU
	cudaMemcpy(C_CPU, C_GPU, sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	DotGPU = C_CPU[0];
	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);
	
	// Check if GPU result matches CPU result within tolerance
	if(!check(DotCPU, DotGPU, Tolerance))
	{
		printf("\n\n Something went wrong in the GPU dot product.\n");
	}
	else
	{
		printf("\n\n You did a dot product correctly on the GPU");
		printf("\n The CPU result is: %f", DotCPU);
		printf("\n The GPU result is: %f", DotGPU);
		printf("\n The time it took on the CPU was %ld microseconds", timeCPU);
		printf("\n The time it took on the GPU was %ld microseconds", timeGPU);
	}
	
	// Cleanup
	CleanUp();	
	
	printf("\n\n");
	
	return 0;
}

