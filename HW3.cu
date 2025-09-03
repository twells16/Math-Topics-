// Name:Tanner Wells
// nvcc HW3.cu -o temp
/*
 What to do:
 This is the solution to HW2. It works well for adding vectors using a single block.
 But why use just one block?
 We have thousands of CUDA cores, so we should use many blocks to keep the SMs (Streaming Multiprocessors) on the GPU busy.

 Extend this code so that, given a block size, it will set the grid size to handle "almost" any vector addition.
 I say "almost" because there is a limit to how many blocks you can use, but this number is very large. 
 We will address this limitation in the next HW.

 Hard-code the block size to be 256.

 Also add cuda error checking into the code.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
#define N 11503 // Length of the vector
													//This gets the last CUDA error and prints out the error, file name, and what line the error happened
													//this then exits the program to avoid any faulty execution
													//this catches any Kernel launch errors(Invalid grid/block size, out-of-bounds memory), memory errors
													//(passing a null pointer, or allocating more memory than allowed), API misuse (Forgetting cudaFree())
													//This is important because this function catches the error immeditaly after it happens instead of allowing 
													//it to continue to run
												
void cudaErrorCheck(const char *file, int line)		
{					
	cudaError_t error;							
	error = cudaGetLastError();

	if(error !=cudaSuccess)
	{
			printf("\n Cuda ERROR: message = %s,File = %s, Line = %d\n",
			cudaGetErrorString(error),file,line);
			exit(0);
	}
}
// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float Tolerance = 0.00000001;

// Function prototypes
void setUpDevices();
void allocateMemory();
void innitialize();
void addVectorsCPU(float*, float*, float*, int);
__global__ void addVectorsGPU(float, float, float, int);
int  check(float*, int);
long elaspedTime(struct timeval, struct timeval);
void cleanUp();

// This will be the layout of the parallel space we will be using.
void setUpDevices()
{
	BlockSize.x = 256;								//Changed the block size to 256 like the intructions asked to do 
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = (N + BlockSize.x- 1) / BlockSize.x;//This helps cover all elements that rest in our vector N, this function also ensures any leftover 
													//elements still get a block but doesn't over/under estimate (uses integer math) this formula is equivalent 
													//to the one in class ((N-B)/B)+1 the fraction is just combined 
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
	cudaMalloc(&A_GPU,N*sizeof(float));				//Use cudaErrorCheck(__FILE__,__LINE__); after every cuda function to properly check if there was an error 
													//and if there is one then 
													//prints what line and what the error is.
	cudaErrorCheck(__FILE__,__LINE__);
	cudaMalloc(&B_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__,__LINE__);
	cudaMalloc(&C_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__,__LINE__);

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
		c[id] = a[id] + b[id];
	}
}

// This is the kernel. It is the function that will run on the GPU.
// It adds vectors a and b on the GPU then stores result in vector c.
__global__ void addVectorsGPU(float *a, float *b, float *c, int n)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(id < n)   								//Because we are now using more than one block the threads need to know what elements to work on and this does that.
												//This also starts over at every 0 block and the threads skip to the previous used blocks 
	{											//The removal of the while loop was because there is no need for the loop to have the threads go to different elements.
												//Since we are operating on more than one block
												//you switch it to an if statement
												//to make the function go all in one go 
		c[id] = a[id] + b[id];
	}
}

// Checking to see if anything went wrong in the vector addition.
int check(float *c, int n)
{
	double sum = 0.0;
	double m = n-1; // Needed the -1 because we start at 0.
	
	for(int id = 0; id < n; id++)
	{ 
		sum += c[id];
	}
	
	if(abs(sum - 3.0*(m*(m+1))/2.0) < Tolerance) 
	{
		return(1);
	}
	else 
	{
		return(0);
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
	cudaErrorCheck(__FILE__,__LINE__);
	cudaFree(B_GPU);
	cudaErrorCheck(__FILE__,__LINE__);
	cudaFree(C_GPU);
	cudaErrorCheck(__FILE__,__LINE__);
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
	cudaErrorCheck(__FILE__,__LINE__);
	cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__,__LINE__);
	
	addVectorsGPU<<<GridSize,BlockSize>>>(A_GPU, B_GPU ,C_GPU, N);
	
	// Copy Memory from GPU to CPU	
	cudaMemcpyAsync(C_CPU, C_GPU, N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__,__LINE__);

	// Making sure the GPU and CPU wiat until each other are at the same place.
	cudaDeviceSynchronize();			//removed "void" because the cudaDeviceSynchronize() does not expect any arguments and thus causing the error
	cudaErrorCheck(__FILE__,__LINE__);
	
	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);
	
	// Checking to see if all went correctly.
	if(check(C_CPU, N) == 0)
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
	printf("\n");
	
	return(0);
}

