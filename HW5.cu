// Name:Tanner Wells
// Device query
// nvcc HW5.cu -o temp
/*
 What to do:
 This code prints out useful information about the GPU(s) in your machine, 
 but there is much more data available in the cudaDeviceProp structure.

 Extend this code so that it prints out all the information about the GPU(s) in your system. 
 Also, and this is the fun part, be prepared to explain what each piece of information means. 
*/

// Include files
#include <stdio.h>

// Defines

// Global variables

// Function prototypes
void cudaErrorCheck(const char*, int);

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

int main()
{
	cudaDeviceProp prop;

	int count;
	cudaGetDeviceCount(&count);
	cudaErrorCheck(__FILE__, __LINE__);
	printf(" You have %d GPUs in this machine\n", count);
	
	for (int i=0; i < count; i++) {
		cudaGetDeviceProperties(&prop, i);
		cudaErrorCheck(__FILE__, __LINE__);
		printf(" ---General Information for device %d ---\n", i);
		printf("Name: %s\n", prop.name);
		printf("Compute capability: %d.%d\n", prop.major, prop.minor);
		printf("Clock rate: %d\n", prop.clockRate);
		printf("Device copy overlap: ");
		if (prop.deviceOverlap) printf("Enabled\n");
		else printf("Disabled\n");
		printf("Kernel execution timeout : ");
		if (prop.kernelExecTimeoutEnabled) printf("Enabled\n");
		else printf("Disabled\n");
		printf(" ---Memory Information for device %d ---\n", i);
		printf("Total global mem: %ld\n", prop.totalGlobalMem);
		printf("Total constant Mem: %ld\n", prop.totalConstMem);
		printf("Max mem pitch: %ld\n", prop.memPitch);
		printf("Texture Alignment: %ld\n", prop.textureAlignment);
		printf(" ---MP Information for device %d ---\n", i);
		printf("Multiprocessor count : %d\n", prop.multiProcessorCount); //Number of SMs on the GPU
		printf("Shared mem per mp: %ld\n", prop.sharedMemPerBlock);
		printf("Registers per mp: %d\n", prop.regsPerBlock);
		printf("Threads in warp: %d\n", prop.warpSize);
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
																																//This is the break on what was added to the code
																																//everything after this code was added 
		
																																
		printf("Integrated: %s\n", (prop.integrated ? "Yes" : "No"));															//This tells you whether the GPU is integrated with 
																																//the CPU if it prints yes then the GPU shares system memory 
																																//if no then the its dedicated to just the GPU
		
		printf("Can Map Host Memory: %s\n", (prop.canMapHostMemory ? "Yes" : "No"));											//This tells you if CUDA can use memory from the CPU 
																																//without having to copy it to the GPU manually 
																																//this reduces the memory copy overhead
		
		printf("Compute Mode: ");
		switch (prop.computeMode) {																							
   			case cudaComputeModeDefault: printf("Default (multiple host threads can use this device)\n"); break;
    		case cudaComputeModeExclusive: printf("Exclusive (only one host thread at a time)\n"); break;
    		case cudaComputeModeProhibited: printf("Prohibited (device not available to any threads)\n"); break;
    		case cudaComputeModeExclusiveProcess: printf("Exclusive Process (only one process at a time)\n"); break;
   			default: printf("Unknown\n"); break;
		}																														//This controls how the GPU is shared across multiple processes
																																//or threads. Default:Multiple threads can use it
																																//Exclusive: Only one thread can use it at a time 
																																//Exclusive Process:Only one process can use it
																																//Prohibited: No access allowed from host 
																																//This is used in server/multi-user enviroments 

		printf("Concurrent Kernels: %s\n", (prop.concurrentKernels ? "Yes" : "No"));											//This checks to see if the GPU can run multiple kernels at once
																																//If yes then it can run multiple tasks in parallel or at the same time
																																//if no then it has to wait until one task is done to move on

		printf("ECC Enabled: %s\n", (prop.ECCEnabled ? "Yes" : "No"));															//This protects against memory bit-flips
																																//this improves the reliability at the cost of performance 

		printf("PCI Bus ID: %d\n", prop.pciBusID);																				
		printf("PCI Device ID: %d\n", prop.pciDeviceID);
		printf("PCI Domain ID: %d\n", prop.pciDomainID);																		//This tells you where the GPU is physically connected on the 
																																//motherboard. This is useful if using multiple GPUs
																																//acts as a street address of the GPU

		printf("Async Engine Count: %d\n", prop.asyncEngineCount);																//This counts how many async engines you have 
																																//This lets you copy memory and run kernels at the same time 
																																//this helps speed up workloads

		printf("Unified Addressing: %s\n", (prop.unifiedAddressing ? "Yes" : "No"));											//This makes it to where the host and device share a single address
																																//space. Can access memory without the cudaMemcpy command

		printf("Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);															
		printf("Memory Bus Width (bits): %d\n", prop.memoryBusWidth);															//This clocks the rate of the GPU's memory in kHz
																																//This tells you how many bits the memory bus can transfer each time 
																																//it runs. Together they determine memory bandwith 

		printf("L2 Cache Size: %d bytes\n", prop.l2CacheSize);																	//Tells you the size of the L2 cache(high speed memory that stores frequently)
																																//shared amoung cores this is used to speed up memory access 

		printf("Max Threads per MultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor);										//This tells you the max number of threads that can run on one Streaming
																																//Multiprocessor 

		printf("Stream Priorities Supported: %s\n", (prop.streamPrioritiesSupported ? "Yes" : "No"));							//This helps you schedule more urgent kernels before others and is 
																																//priority levels supported for CUDA streams 

		printf("Global L1 Cache Supported: %s\n", (prop.globalL1CacheSupported ? "Yes" : "No"));								//
		printf("Local L1 Cache Supported: %s\n", (prop.localL1CacheSupported ? "Yes" : "No"));
		printf("Host Native Atomic Supported: %s\n", (prop.hostNativeAtomicSupported ? "Yes" : "No"));
		printf("Single to Double Precision Perf Ratio: %d\n", prop.singleToDoublePrecisionPerfRatio);
		printf("Pageable Memory Access: %s\n", (prop.pageableMemoryAccess ? "Yes" : "No"));	
		printf("Pageable Memory Access Uses Host Page Tables: %s\n", (prop.pageableMemoryAccessUsesHostPageTables ? "Yes" : "No"));
		printf("Direct Managed Memory Access From Host: %s\n", (prop.directManagedMemAccessFromHost ? "Yes" : "No"));
		printf("Concurrent Managed Access: %s\n", (prop.concurrentManagedAccess ? "Yes" : "No"));
		printf("Compute Preemption Supported: %s\n", (prop.computePreemptionSupported ? "Yes" : "No"));
		printf("Can Use Host Pointer For Registered Mem: %s\n", (prop.canUseHostPointerForRegisteredMem ? "Yes" : "No"));

		printf("\n");
	}	
	return(0);
}

