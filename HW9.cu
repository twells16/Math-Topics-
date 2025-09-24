// Name:Tanner Wells
// Vector Dot product on many blocks using shared memory
// nvcc HW9.cu -o temp
/*
 What to do:
 This code is the solution to HW8. It finds the dot product of vectors that are smaller than the block size.
 Extend this code so that it sets as many blocks as needed for a set thread count and vector length.
 Use shared memory in your blocks to speed up your code.
 You will have to do the final reduction on the CPU.
 Set your thread count to 200 (block size = 200). Set N to different values to check your code.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
#define N 2500 // Length of the vector
#define BLOCK_SIZE 200 // This defines the thread count

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
void dotProductCPU(float*, float*, int);
__global__ void dotProductGPU(float*, float*, float*, int);
bool  check(float, float, float);
long elaspedTime(struct timeval, struct timeval);
void cleanUp();

// This check to see if an error happened in your CUDA code. It tell you what it thinks went wrong,
// and what file and line it occured on.
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
	BlockSize.x = BLOCK_SIZE;                                     			//We defined ho wbig the block was above 
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	// FIXED: Calculate number of blocks needed to cover all elements
	GridSize.x = (N + BLOCK_SIZE - 1) / BLOCK_SIZE; 						// Ceiling division calculates the block size
	GridSize.y = 1;
	GridSize.z = 1;
	
	//printf("Setup: %d elements, %d blocks x %d threads = %d total threads\n", //This line shopwed me during the debugging stage the setup I had is no longer needed 
	     //N, GridSize.x, BLOCK_SIZE, GridSize.x * BLOCK_SIZE);					as teh problem was fixed. 
}

// FIXED: Allocating memory for partial results from each block
void allocateMemory()
{	
	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(GridSize.x*sizeof(float)); // CRITICAL: One result per block
	
	// Device "GPU" Memory
	cudaMalloc(&A_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU,GridSize.x*sizeof(float)); // CRITICAL: One result per block
	cudaErrorCheck(__FILE__, __LINE__);
}

// Loading values into the vectors that we will add.
void innitialize()
{
	for(int i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)(3*i);
	}
}

// FIXED: Proper CPU dot product calculation
void dotProductCPU(float *a, float *b, int n)
{
	DotCPU = 0.0f;
	for(int i = 0; i < n; i++)
	{ 
		DotCPU += a[i] * b[i];
	}
}

// COMPLETELY REWRITTEN: Multi-block GPU kernel with shared memory
__global__ void dotProductGPU(float *a, float *b, float *c, int n)
{
	// REQUIREMENT: Use shared memory for speed
	extern __shared__ float shared_data[];								//This dynamic shared memory and is used to store per-thread dot products 
																		//within the block 
	
	// CRITICAL FIX: Use global thread ID across ALL blocks
	int global_tid = blockIdx.x * blockDim.x + threadIdx.x;				//Thread index across the entore grid 
	int local_tid = threadIdx.x;										//Thread index within each block 
	
	// Initialize shared memory with boundary checking					//This is where each thread computes the dot product and stores it in the shared memory 
																		//added a boundary check to ensure tehres no access out-of-bounds memory if N is not divisble by BLOCK_SIZE
	if (global_tid < n) {
		shared_data[local_tid] = a[global_tid] * b[global_tid];
	} else {
		shared_data[local_tid] = 0.0f; // Pad with zeros for out-of-bounds threads
	}
	
	__syncthreads();													//Ensures all of the threads sync together before moving on to the next part
	
	// FIXED: Most robust reduction for non-power-of-2 block sizes		//This performs a binary reduction tree and the adds onto the next-level
																		//This then takes the final result from the tree and loads it to teh shared memory
	for (int stride = 1; stride < blockDim.x; stride *= 2) {
		if ((local_tid % (2 * stride)) == 0) {
			if (local_tid + stride < blockDim.x) {
				shared_data[local_tid] += shared_data[local_tid + stride];
			}
		}
		__syncthreads();
	}
	
	// Thread 0 of each block writes the partial result to global memory	//Only thread 0 of each block writes teh block's final partial sum to global memory  
	if (local_tid == 0) {
		c[blockIdx.x] = shared_data[0];
	}
}

// FIXED: Proper error checking with division by zero handling
bool check(float cpuAnswer, float gpuAnswer, float tolerence)
{
	double percentError;
	
	if (cpuAnswer == 0.0f) {
		percentError = abs(gpuAnswer) * 100.0;
	} else {
		percentError = abs((gpuAnswer - cpuAnswer)/(cpuAnswer))*100.0;
	}
	
	printf("\nCPU Result: %.2f", cpuAnswer);
	printf("\nGPU Result: %.2f", gpuAnswer);
	printf("\nPercent error = %.6lf%%\n", percentError);
	
	if(percentError < tolerence) 
	{
		return(true);
	}
	else 
	{
		return(false);
	}
}

// Calculating elapsed time.
long elaspedTime(struct timeval start, struct timeval end)
{
	long startTime = start.tv_sec * 1000000 + start.tv_usec;
	long endTime = end.tv_sec * 1000000 + end.tv_usec;
	return endTime - startTime;
}

// Cleaning up memory after we are finished.
void cleanUp()
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
	
	// Setting up the GPU
	setUpDevices();
	
	// Allocating the memory you will need.
	allocateMemory();
	
	// Putting values in the vectors.
	innitialize();
	
	// Computing dot product on the CPU
	gettimeofday(&start, NULL);
	dotProductCPU(A_CPU, B_CPU, N);
	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);
	
	// REMOVED: The problematic check that always exits
	// The whole point is to handle vectors larger than block size!
	
	// Computing dot product on the GPU
	gettimeofday(&start, NULL);
	
	// Copy Memory from CPU to GPU		
	cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// REQUIREMENT: Launch kernel with shared memory
	int shared_mem_size = BLOCK_SIZE * sizeof(float);
	dotProductGPU<<<GridSize,BlockSize,shared_mem_size>>>(A_GPU, B_GPU, C_GPU, N);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// FIXED: Copy ALL partial results from GPU to CPU	
	cudaMemcpyAsync(C_CPU, C_GPU, GridSize.x*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Making sure the GPU and CPU wait until each other are at the same place.
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);

	// REQUIREMENT: Final reduction on CPU
	DotGPU = 0.0f;
	printf("\nPartial results from each block:\n");									//Prints each result from each block helped during the debugging proccess
	for (int i = 0; i < GridSize.x; i++) {
		printf("Block %d: %.2f\n", i, C_CPU[i]);
		DotGPU += C_CPU[i];
	}

	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);
	
	// Checking to see if all went correctly.
	if(check(DotCPU, DotGPU, Tolerance) == false)
	{
		printf("\nSomething went wrong in the GPU dot product.\n");
	}
	else
	{
		printf("\nYou did a dot product correctly on the GPU");
		printf("\nThe time it took on the CPU was %ld microseconds", timeCPU);
		printf("\nThe time it took on the GPU was %ld microseconds", timeGPU);
	}
	
	// Your done so cleanup your room.	
	cleanUp();	
	
	// Making sure it flushes out anything in the print buffer.
	printf("\n\n");
	
	return(0);
}