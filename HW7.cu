// Name:Tanner Wells 
// Not simple Julia Set on the GPU
// nvcc HW7.cu -o temp -lglut -lGL

/*
 What to do:
 This code displays a simple Julia set fractal using the GPU.
 But it only runs on a window of 1024X1024.
 Extend it so that it can run on any given window size.
 Also, color it to your liking. I will judge you on your artisct flare. 
 Don't cute off your ear or anything but make Vincent wish he had, had a GPU.
*/

// Include files
#include <stdio.h>
#include <GL/glut.h>
#include <stdlib.h>
#include <math.h>						//allows us to use the math functions 

// Defines
#define MAXMAG 10.0 // If you grow larger than this, we assume that you have escaped.
#define MAXITERATIONS 200 // If you have not escaped after this many attempts, we assume you are not going to escape.
#define A  -0.824	//Real part of C
#define B  -0.1711	//Imaginary part of C

// Global variables
unsigned int WindowWidth = 1024;
unsigned int WindowHeight = 1024;

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;

// Function prototypes
void cudaErrorCheck(const char*, int);
__global__ void colorPixels(float, float, float, float, float);
//This function checks for any CUDA errors and returns the line on where its happening and helps 
//with debugging your code
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

//Rewrote the colorPixels so that this could allow for any window size this is also where the GPU does all the heavy fractal math
//in parallel
__global__ void colorPixels(float *pixels, float xMin, float yMin, float dx, float dy, int windowWidth, int windowHeight) 
{
	int pixelX = threadIdx.x + blockIdx.x * blockDim.x;  									
	int pixelY = threadIdx.y + blockIdx.y * blockDim.y;  									
	int pixelIndex = pixelY * windowWidth + pixelX;

	if (pixelX >= windowWidth || pixelY >= windowHeight)
    return;


	int id = 3 * pixelIndex;

	float x = xMin + dx * pixelX;
	float y = yMin + dy * pixelY;
	float tempX;
	int count = 0;
	float mag = sqrtf(x * x + y * y);
//This loop applies the Julia Set formula and counts the iterations for it to escape the bounds
	while (mag < MAXMAG && count < MAXITERATIONS)
	{
		tempX = x;
		x = x * x - y * y + A;
		y = (2.0f * tempX * y) + B;
		mag = sqrtf(x * x + y * y);
		count++;
	}

	float t = (float)count / MAXITERATIONS;
	float fx = (float)pixelX / (gridDim.x * blockDim.x);
	float fy = (float)pixelY / (gridDim.y * blockDim.y);
//This gives the weird color scheme based on the how quickly a point escaped and its position 
	pixels[id    ] = 0.5f + 0.5f * cosf(10.0f * t + fx * 8.0f);   // Red, also where you see our math file that we included come into play
	pixels[id + 1] = 0.5f + 0.5f * sinf(15.0f * t + fy * 5.0f);   // Green
	pixels[id + 2] = 0.5f + 0.5f * cosf(25.0f * t + fx * fy * 6); // Blue
}

//This calculates the step size for each pixel in the coordinate system. This also allocates memory on our CPU as well as the GPU using the pixel command
//This sets up the CUDA thread blocks anbd grid, colorPixels then calls the CUDA kernel and computes the fractal color for each pixel
//It then takes the data from the GPU and copies back to the CPU, and draws the the computed pixels on the OpenGL window.
//At the end it frees up all the allocated memory on the CPU and GPU this helps avoid memory leaks
void display(void) 
{ 
	float *pixelsCPU, *pixelsGPU;
	float stepSizeX = (XMax - XMin) / (float)WindowWidth;
	float stepSizeY = (YMax - YMin) / (float)WindowHeight;

	size_t bufferSize = WindowWidth * WindowHeight * 3 * sizeof(float);
	pixelsCPU = (float *)malloc(bufferSize);
	cudaMalloc(&pixelsGPU, bufferSize);
	cudaErrorCheck(__FILE__, __LINE__);

	//NEW: dynamically set up blocks & grid to support any window size
	dim3 blockSize(16, 16);
	dim3 gridSize(
		(WindowWidth + blockSize.x - 1) / blockSize.x,
		(WindowHeight + blockSize.y - 1) / blockSize.y
	);

	colorPixels<<<gridSize, blockSize>>>(pixelsGPU, XMin, YMin, stepSizeX, stepSizeY,WindowWidth, WindowHeight);
	cudaErrorCheck(__FILE__, __LINE__);

	cudaMemcpy(pixelsCPU, pixelsGPU, bufferSize, cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
    //This makes sure it clears any old drawings 
    glClear(GL_COLOR_BUFFER_BIT);

	glDrawPixels(WindowWidth, WindowHeight, GL_RGB, GL_FLOAT, pixelsCPU);
	glFlush();

	cudaFree(pixelsGPU); // This frees up the memory and pixels on the GPU
	free(pixelsCPU);     // This frees up memory and the pixels 
}
//This function allows the code to resize and makes sure the code doesnt break when it is resized
void reshape(int w, int h)
{
    WindowWidth = w;
    WindowHeight = h;

    glViewport(0, 0, w, h);

    float aspectRatio = (float)w / (float)h;
    float baseSize = 4.0f;

    if (aspectRatio >= 1.0f)
    {
        XMin = -baseSize * aspectRatio / 2.0f;
        XMax = baseSize * aspectRatio / 2.0f;
        YMin = -baseSize / 2.0f;
        YMax = baseSize / 2.0f;
    }
    else
    {
        XMin = -baseSize / 2.0f;
        XMax = baseSize / 2.0f;
        YMin = -baseSize / aspectRatio / 2.0f;
        YMax = baseSize / aspectRatio / 2.0f;
    }
glutPostRedisplay();
}
//This function initalizes the GLUT toolkit and sets the display mode for our fun colors. This also creates the window with the window size
//It then registers the callback functions display which allows the window to be redrawn and the reshape function for when we need to resize that 
//window. This then starts the GLUT loop 
int main(int argc, char** argv)
{

	if (argc == 3)
	{
		int w = atoi(argv[1]);
		int h = atoi(argv[2]);
		if (w > 0 && h > 0)
		{
			WindowWidth = w;
			WindowHeight = h;
		}
		else
		{
			printf("Invalid size. Using default 1024x1024.\n");
		}
	}
	else if (argc != 1)
	{
		printf("Usage: %s [width height]\n", argv[0]);
		return 1;
	}

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
	glutInitWindowSize(WindowWidth, WindowHeight);
	glutCreateWindow(" Psychedelic GPU Julia Set");

	glutDisplayFunc(display);
	glutReshapeFunc(reshape); //
	glutMainLoop();
}



