// Name: Tanner Wells 
// GPU random walk. 
// nvcc 16GPURandomWalk.cu -o temp

/*
 What to do:
 This code runs a random walk for 10,000 steps on the CPU.

 1. Use cuRAND to run 20 random walks simultaneously on the GPU, each with a different seed.
    Print out all 20 final positions.

 2. Use cudaMallocManaged(&variable, amount_of_memory_needed);
    This allocates unified memory, which is automatically managed between the CPU and GPU.
    You lose some control over placement, but it saves you from having to manually copy data
    to and from the GPU.
*/

/*
 Purpose:
 To learn how to use cuRAND and unified memory.
*/

/*
 Note:
 The maximum signed int value is 2,147,483,647, so the maximum unsigned int value is 4,294,967,295.

 RAND_MAX is guaranteed to be at least 32,767. When I checked it on my laptop (10/6/2025), it was 2,147,483,647.
 rand() returns a value in [0, RAND_MAX]. It actually generates a list of pseudo-random numbers that depends on the seed.
 This list eventually repeats (this is called its period). The period is usually 2³¹ = 2,147,483,648,
 but it may vary by implementation.

 Because RAND_MAX is odd on this machine and 0 is included, there is no exact middle integer.
 Casting to float as in (float)RAND_MAX / 2.0 divides the range evenly.
 Using integer division (RAND_MAX / 2) would bias results slightly toward the positive side by one value out of 2,147,483,647.

 I know this is splitting hares (sorry, rabbits), but I'm just trying to be as accurate as possible.
 You might do this faster with a clever integer approach, but I’m using floats here for clarity.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>
#include <curand_kernel.h>//This include file allows for randome numbers to be generated directly within the a CUDA file 
//#include <cuda_runtime.h> //Allows us to use functions for cuda memory and kernel execution 
#include <time.h>//This allows us to take the current time of day when we make our seed 

// Defines
#define NUM_WALKS 20
#define NUM_STEPS 10000

// Globals
int NumberOfRandomSteps = 10000;
float MidPoint = (float)RAND_MAX/2.0f;

// Function prototypes
int getRandomDirection();
int main(int, char**);

/*GPU Kernel that will run our drunk walk. 
This defines our variables for our X, and Y positions as well as 
our seed that is whole positive numbers, This then makes our unique thread id,
we then define the cudaranState that defines our random numbers in state,
we then set a random generator and sets it parameters,
we then generate a randome float between 0.0-1.0,
we then use an if/else statement to check where the randome float is
*/
__global__ void randomWalk(int *positionsX, int *positionsY, unsigned long long seed)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid>= NUM_WALKS) return;

	curandState state;
	curand_init(seed + tid, 0, 0, &state);

	int posX = 0;
	int posY = 0;

	for (int i = 0; i < NUM_STEPS; i++)
	{
		float randX = curand_uniform(&state);
		float randY = curand_uniform(&state);

		if (randX < 0.5f)
    	posX += -1;
		else
  		posX += 1;

		if (randY < 0.5f)
		posY += -1;
		else
		posY +=1;
	}

	positionsX[tid] = posX;
	positionsY[tid] = posY;

}

/* This grabs what direction our person will move
using the rand function*/
int getRandomDirection()
{	
	int randomNumber = rand();
	
	if(randomNumber < MidPoint) return(-1);
	else return(1);
}

int main(int argc, char** argv)
{
	int *positionsX, *positionsY;

	//allows automatic transfer of data between the host and device
	//and allows for unified data access from the CPU and GPU
	cudaMallocManaged(&positionsX, NUM_WALKS * sizeof(int));
	cudaMallocManaged(&positionsY, NUM_WALKS * sizeof(int));
	//This sets the number of threads per block and finds how many blocks you will use
	int threadsPerBlock = 20;
	int blocks = (NUM_WALKS + threadsPerBlock - 1) / threadsPerBlock;
	//this uses time of day to make your seed
	unsigned long long seed = time(NULL);
	//this calls the kernel you made above
	randomWalk<<<blocks, threadsPerBlock>>>(positionsX, positionsY, seed);
	//This syncs the host and the device 
	cudaDeviceSynchronize();
	/*Removed the CPU code*/
	//srand(time(NULL));
	
	printf("RAND_MAX for this implementation is = %d \n", RAND_MAX);
	/*Removed the CPU function*/
	/*
	int positionX = 0;
	int positionY = 0;
	for(int i = 0; i < NumberOfRandomSteps; i++)
	{
		positionX += getRandomDirection();
		positionY += getRandomDirection();
	}
	
	printf("\n Final position = (%d,%d) \n", positionX, positionY);
	*/
	/*This prints each walk and its position as well the final position of our drunk person*/
	printf("\nFinal Positions for %d walks : \n", NUM_WALKS);
	for (int i = 0; i < NUM_WALKS; i++)
	{
		printf("Walk %2d: (%d, %d)\n", i, positionsX[i], positionsY[i]);
	}


	//This frees our memory from the GPU that we stored.
	cudaFree(positionsX);
	cudaFree(positionsY);
	return 0;
}


