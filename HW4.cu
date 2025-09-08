// Name: Tanner Wells
// Vector addition on the GPU of any size with fixed block and grid size also adding pragma unroll for speed up.
// nvcc HW4.cu -o temp
/*
 What to do:
 This is the solution to HW3. It works well for adding vectors with fixed-size blocks. 
 Given the size of the vector it needs to add, it takes a set block size, determines how 
 many blocks are needed, and creates a grid large enough to complete the task. Cool, cool!
 
 But—and this is a big but—this can get you into trouble because there is a limited number 
 of blocks you can use. Though large, it is still finite. Therefore, we need to write the 
 code in such a way that we don't have to worry about this limit. Additionally, some block 
 and grid sizes work better than others, which we will explore when we look at the 
 streaming multiprocessors.
 
 Extend this code so that, given a block size and a grid size, it can handle any vector addition. 
 Start by hard-coding the block size to 256 and the grid size to 64. Then, experiment with different 
 block and grid sizes to see if you can achieve any speedup. Set the vector size to a very large value 
 for time testing.

 You’ve probably already noticed that the GPU doesn’t significantly outperform the CPU. This is because 
 we’re not asking the GPU to do much work, and the overhead of setting up the GPU eliminates much of the 
 potential speedup. 
 
 To address this, modify the computation so that:
 c = sqrt(cos(a)*cos(a) + a*a + sin(a)*sin(a) - 1.0) + sqrt(cos(b)*cos(b) + b*b + sin(b)*sin(b) - 1.0)
 Hopefully, this is just a convoluted and computationally expensive way to calculate a + b.
 If the compiler doesn't recognize the simplification and optimize away all the unnecessary work, 
 this should create enough computational workload for the GPU to outperform the CPU.

 Write the loop as a for loop rather than a while loop. This will allow you to also use #pragma unroll 
 to explore whether it provides any speedup. Make sure to include an if (id < n) condition in your code 
 to ensure safety. Finally, be prepared to discuss the impact of #pragma unroll and whether it helped 
 improve performance.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
#define N 100000000 // Length of the vector                                                     //Change the length of the vector to be a very large number to test the GPU performace under
                                                                                                //a heavier load

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float Tolerance = 0.01;

// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices();
void allocateMemory();
void innitialize();
void addVectorsCPU(float*, float*, float*, int);
__global__ void addVectorsGPU(float, float, float, int);
bool  check(float*, int);
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
	BlockSize.x = 256;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = 64; // This gives us the correct number of blocks.                     //Hard coded the GridSize.x to 64 like the instructions asked us to do 
	GridSize.y = 1;
	GridSize.z = 1;
}

// Allocating the memory we will be using.
void allocateMemory()
{	
	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
	
	// Device "GPU" Memory
	cudaMalloc(&A_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU,N*sizeof(float));
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

// Adding vectors a and b on the CPU then stores result in vector c.
void addVectorsCPU(float *a, float *b, float *c, int n)
{
	for(int id = 0; id < n; id++)
	{ 
		c[id] = sqrtf(cosf(a[id]) * cosf(a[id]) + a[id]*a[id] + sinf(a[id]) * sinf(a[id]) - 1.0f)      //Changed the a + b equation to this one to have more computation on the CPU
                                                                                                       //the way this adds more work is its doing more work on the kernel going back
                                                                                                       //and forth between the CPU and the GPU. This causes more time on the GPU and CPU 
                                                                                                       //doing math. This also helps reveal the advantages of parallel programming.
              + sqrtf(cosf(b[id])* cosf(b[id]) + b[id]*b[id] + sinf(b[id]) * sinf(b[id]) - 1.0f);
	}
}

// This is the kernel. It is the function that will run on the GPU.
// It adds vectors a and b on the GPU then stores result in vector c.
__global__ void addVectorsGPU(float *a, float *b, float *c, int n)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	#pragma unroll                                                                                      //This enables support for any vector size, and helps with performance tuning. 
                                                                                                        //This function expands the loop so that fewer iterations and steps are taken.
                                                                                                        //This boost the performance by elimination loop overhead and allows the GPU
                                                                                                        //to execute more instructions when unrolled thus speeding up the GPU a little 
                                                                                                        //and out performs the CPU.
	for (int i = id; i < n; i += blockDim.x * gridDim.x)
    {
        if (i < n)                                                                                      //This equation ensures that the CPU and GPU are doing the same work for the same reason
                                                                                                        //stated above.
        {
            c[i] = sqrtf(cosf(a[i]) * cosf(a[i]) + a[i]*a[i] + sinf(a[i]) * sinf(a[i]) - 1.0f)
                 + sqrtf(cosf(b[i]) * cosf(b[i]) + b[i]*b[i] + sinf(b[i]) * sinf(b[i]) - 1.0f);
        }
    }
}

// Checking to see if anything went wrong in the vector addition.
bool check(float *c, int n, float tolerence)
{
	int id;
	double myAnswer;
	double trueAnswer;
	double percentError;
	double m = n-1; // Needed the -1 because we start at 0.
	
	myAnswer = 0.0;
	for(id = 0; id < n; id++)
	{ 
		myAnswer += c[id];
	}
	
	trueAnswer = 3.0*(m*(m+1))/2.0;
	
	percentError = abs((myAnswer - trueAnswer)/trueAnswer)*100.0;
	
	if(percentError < Tolerance) 
	{
		return(true);
	}
	else 
	{
		return(false);
	}
}

// Calculating elasped time.
long elaspedTime(struct timeval start, struct timeval end)
{
	// tv_sec = number of seconds past the Unix epoch 01/01/1970
	// tv_usec = number of microseconds past the current second.
	
	long startTime = start.tv_sec * 1000000 + start.tv_usec; // In microseconds.
	long endTime = end.tv_sec * 1000000 + end.tv_usec; // In microseconds

	// Returning the total time elasped in microseconds
	return endTime - startTime;
}

// Cleaning up memory after we are finished.
void CleanUp()
{
	// Freeing host "CPU" memory.
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
	
	// Adding on the CPU
	gettimeofday(&start, NULL);
	addVectorsCPU(A_CPU, B_CPU ,C_CPU, N);
	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);
	
	// Zeroing out the C_CPU vector just to be safe because right now it has the correct answer in it.
	for(int id = 0; id < N; id++)
	{ 
		C_CPU[id] = 0.0;
	}
	
	// Adding on the GPU
	gettimeofday(&start, NULL);
	
	// Copy Memory from CPU to GPU		
	cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	addVectorsGPU<<<GridSize,BlockSize>>>(A_GPU, B_GPU ,C_GPU, N);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Copy Memory from GPU to CPU	
	cudaMemcpyAsync(C_CPU, C_GPU, N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Making sure the GPU and CPU wiat until each other are at the same place.
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);
	
	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);
	
	// Checking to see if all went correctly.
	if(check(C_CPU, N, Tolerance) == false)
	{
		printf("\n\n Something went wrong in the GPU vector addition\n");
	}
	else
	{
		printf("\n\n You added the two vectors correctly on the GPU");
		printf("\n The time it took on the CPU was %ld microseconds", timeCPU);
		printf("\n The time it took on the GPU was %ld microseconds", timeGPU);
	}
	
	// Your done so cleanup your room.	
	CleanUp();	
	
	// Making sure it flushes out anything in the print buffer.
	printf("\n\n");
	
	return(0);
}
