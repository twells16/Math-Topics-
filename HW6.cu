// Name:Tanner Wells
// Simple Julia CPU.
// nvcc HW6.cu -o temp -lglut -lGL
// glut and GL are openGL libraries.
/*
 What to do:
 This code displays a simple Julia fractal using the CPU.
 Rewrite the code so that it uses the GPU to create the fractal. 
 Keep the window at 1024 by 1024.
*/

// Include files
#include <stdio.h>
#include <GL/glut.h>
#include <cuda_runtime.h>                                                //This allows you to use CUDA functions such as cudaMalloc, and cudaMempy as well as kernel launching

// Defines
#define MAXMAG 10.0 // If you grow larger than this, we assume that you have escaped.
#define MAXITERATIONS 200 // If you have not escaped after this many attempts, we assume you are not going to escape.
#define A  -0.824	//Real part of C
#define B  -0.1711	//Imaginary part of C
#define cudaCheckError() { cudaError_t e=cudaGetLastError(); if(e!=cudaSuccess) { \
    printf("CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); }}		//This macro checks for any CUDA errors and prints details if there was


// Global variables
unsigned int WindowWidth = 1024;
unsigned int WindowHeight = 1024;

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;

float *d_pixels = nullptr;												//this is the pointer to GPU memory where the CUDA kernel writes the pixel colors
float *h_pixels = nullptr;												//this is a buffer where pixels are copied after GPU computation to display 


// Function prototypes
void cudaErrorCheck(const char*, int);
float escapeOrNotColor(float, float);

__global__ void computeJulia(float* pixels, int width, int height, float XMin, float XMax, float YMin, float YMax)

{																								//This kernel replaces the CPU function that ran loops and replaced it with the a function
																								//that runs thousands of threads in parallel on the GPU with each thread computing a pixel
    int xIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int yIdx = blockIdx.y * blockDim.y + threadIdx.y;

    if(xIdx >= width || yIdx >= height)
        return;

    float stepX = (XMax - XMin) / width;
    float stepY = (YMax - YMin) / height;

    float zx = XMin + xIdx * stepX;
    float zy = YMin + yIdx * stepY;

    int count = 0;
    while (zx * zx + zy * zy < MAXMAG * MAXMAG && count < MAXITERATIONS)
    {
        float tmp = zx * zx - zy * zy + A;
        zy = 2.0f * zx * zy + B;
        zx = tmp;
        count++;
    }

    float color = (count == MAXITERATIONS) ? 1.0f : 0.0f;

    int idx = 3 * (yIdx * width + xIdx);
    pixels[idx] = color;     // Red
    pixels[idx + 1] = 0.0f;  // Green
    pixels[idx + 2] = 0.0f;  // Blue
}
void display(void) 
{
    // Launch CUDA kernel
    dim3 blockSize(16, 16);																				//This now launches a GPU kernel, after this kernel finishes it copies the GPU 
																										//pixel data back to teh CPU to display using OpenGL
    dim3 gridSize( (WindowWidth + blockSize.x - 1) / blockSize.x,
                   (WindowHeight + blockSize.y - 1) / blockSize.y );

    computeJulia<<<gridSize, blockSize>>>(d_pixels, WindowWidth, WindowHeight, XMin, XMax, YMin, YMax);
    cudaDeviceSynchronize();
    cudaCheckError();

    // Copy results back to host
    cudaMemcpy(h_pixels, d_pixels, WindowWidth * WindowHeight * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();

    // Display using OpenGL
    glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, h_pixels);
    glFlush();
}

int main(int argc, char** argv)
{ 																				
   h_pixels = (float*)malloc(WindowWidth * WindowHeight * 3 * sizeof(float));		//This was needed so we can receive pixel data after the GPU computation for OpenGL display h_pixels
																					//d_pixels is for the GPU kernel to write pixel data 
    if (!h_pixels)
    {
        fprintf(stderr, "Failed to allocate host pixel buffer\n");
        return -1;
    }

    cudaMalloc(&d_pixels, WindowWidth * WindowHeight * 3 * sizeof(float));
    cudaCheckError();

    // Initialize GLUT window														//This allowed OpenGL window setup and display callback to keep the same window and rendering style
																					//This set ups a OpenGL and enters the maun event loop where the display() is called
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
    glutInitWindowSize(WindowWidth, WindowHeight);
    glutCreateWindow("Julia Fractal GPU");
    glutDisplayFunc(display);

    glutMainLoop();

    // Cleanup (unreachable here but good practice)
    cudaFree(d_pixels);
    free(h_pixels);

    return 0;
}

