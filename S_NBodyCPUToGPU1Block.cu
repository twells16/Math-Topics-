// Name:Tanner Wells (GPU offload version)
// S_NBodyGPUOneBlock.cu
//
// nvcc NBodyCPUToGPUTest.cu -o temp -lglut -lm -lGLU -lGL
//
// This version offloads the compute-intensive nBody loop to the GPU.
// It keeps the same overall program structure and drawing logic.
// Constraints: no ternary operators, no compound assignment operators (use a = a + b etc).

#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

// Defines
#define PI 3.14159265359f
#define DRAW_RATE 10

// Lennard-Jones-like force constants (p=2, q=4)
#define G 10.0f
#define H 10.0f
#define LJP 2.0f
#define LJQ 4.0f

#define DT 0.0001f
#define RUN_TIME 1.0f

// Globals (host)
int N, DrawFlag;
float3 *h_P, *h_V;
float3 *h_F; // host scratch (not strictly required but kept for parity)
float *h_M;
float GlobeRadius, Diameter, Radius;
float Damp;

// Device pointers
float3 *d_P;
float3 *d_V;
float *d_M;

// Prototypes
void keyPressed(unsigned char, int, int);
long elapsedTime(struct timeval, struct timeval);
void drawPicture();
void timer();
void setup();
void nBody(); // host wrapper that uses device kernels
int main(int, char**);

// Simple error check macro (no compound ops used)
#define CUDA_CHECK(err) do { \
    cudaError_t errCode = (err); \
    if(errCode != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (line %d)\n", cudaGetErrorString(errCode), __LINE__); \
        exit(1); \
    } \
} while(0)

void keyPressed(unsigned char key, int x, int y)
{
    if(key == 's')
    {
        timer();
    }

    if(key == 'q')
    {
        exit(0);
    }
}

long elapsedTime(struct timeval start, struct timeval end)
{
    long startTime = start.tv_sec * 1000000L + start.tv_usec;
    long endTime = end.tv_sec * 1000000L + end.tv_usec;
    return endTime - startTime;
}

void drawPicture()
{
    int i;
    glClear(GL_COLOR_BUFFER_BIT);
    glClear(GL_DEPTH_BUFFER_BIT);

    glColor3d(1.0,1.0,0.5);
    for(i = 0; i < N; i = i + 1)
    {
        glPushMatrix();
        glTranslatef(h_P[i].x, h_P[i].y, h_P[i].z);
        glutSolidSphere(Radius,20,20);
        glPopMatrix();
    }

    glutSwapBuffers();
}

// GPU kernel:
// For each body i (one thread per i), compute net force due to all j (j != i)
// Then update velocity and position for one time step dt.
// isFirstStep toggles the half-step initialization from the original code.
__global__ void nbody_kernel(float3 *P, float3 *V, float *M, int N_local, float dt_local, float Damp_local, int isFirstStep)
{
    int tid = threadIdx.x;
    if(tid >= N_local)
    {
        return;
    }

    float fx = 0.0f;
    float fy = 0.0f;
    float fz = 0.0f;

    // read position of this body
    float xi = P[tid].x;
    float yi = P[tid].y;
    float zi = P[tid].z;

    float mi = M[tid];

    // Sum forces from all other bodies
    int j;
    for(j = 0; j < N_local; j = j + 1)
    {
        if(j == tid)
        {
            // skip self
        }
        else
        {
            float dx = P[j].x - xi;
            float dy = P[j].y - yi;
            float dz = P[j].z - zi;
            float d2 = dx*dx + dy*dy + dz*dz;
            float d = sqrtf(d2);

            // compute force magnitude: (G*m_i*m_j)/d^2 - (H*m_i*m_j)/d^4
            float mj = M[j];
            float common = mi * mj;
            float term1 = (G * common) / d2;
            float d4 = d2 * d2;
            float term2 = (H * common) / d4;
            float force_mag = term1 - term2;

            // add contribution to force vector (force_mag * r_hat)
            // avoid compound ops: fx = fx + ...
            fx = fx + force_mag * dx / d;
            fy = fy + force_mag * dy / d;
            fz = fz + force_mag * dz / d;
        }
    }

    // Update velocity and position.
    float vx = V[tid].x;
    float vy = V[tid].y;
    float vz = V[tid].z;

    if(isFirstStep == 1)
    {
        // V += (F/M)*0.5*dt
        float vx_new = vx + (fx / mi) * 0.5f * dt_local;
        float vy_new = vy + (fy / mi) * 0.5f * dt_local;
        float vz_new = vz + (fz / mi) * 0.5f * dt_local;
        vx = vx_new;
        vy = vy_new;
        vz = vz_new;
    }
    else
    {
        // V += ((F - Damp*V)/M)*dt
        float fx_damped = fx - Damp_local * vx;
        float fy_damped = fy - Damp_local * vy;
        float fz_damped = fz - Damp_local * vz;

        float vx_new = vx + (fx_damped / mi) * dt_local;
        float vy_new = vy + (fy_damped / mi) * dt_local;
        float vz_new = vz + (fz_damped / mi) * dt_local;
        vx = vx_new;
        vy = vy_new;
        vz = vz_new;
    }

    // Position update: P += V * dt
    float x_new = xi + vx * dt_local;
    float y_new = yi + vy * dt_local;
    float z_new = zi + vz * dt_local;

    // write back
    V[tid].x = vx;
    V[tid].y = vy;
    V[tid].z = vz;

    P[tid].x = x_new;
    P[tid].y = y_new;
    P[tid].z = z_new;
}

void timer()
{
    struct timeval start, end;
    long computeTime;

    drawPicture();
    gettimeofday(&start, NULL);
    nBody();
    gettimeofday(&end, NULL);
    drawPicture();

    computeTime = elapsedTime(start, end);
    printf("\n The compute time was %ld microseconds.\n\n", computeTime);
}

void setup()
{
    float randomAngle1, randomAngle2, randomRadius;
    float d, dx, dy, dz;
    int test;

    Damp = 0.5f;

    h_M = (float*)malloc(N * sizeof(float));
    h_P = (float3*)malloc(N * sizeof(float3));
    h_V = (float3*)malloc(N * sizeof(float3));
    h_F = (float3*)malloc(N * sizeof(float3)); // not heavily used on host but kept

    Diameter = powf(H / G, 1.0f / (LJQ - LJP));
    Radius = Diameter / 2.0f;

    float totalVolume = (float)N * (4.0f/3.0f) * PI * Radius * Radius * Radius;
    totalVolume = totalVolume / 0.68f;
    float totalRadius = powf(3.0f * totalVolume / (4.0f * PI), 1.0f/3.0f);
    GlobeRadius = 2.0f * totalRadius;

    int i;
    for(i = 0; i < N; i = i + 1)
    {
        test = 0;
        while(test == 0)
        {
            randomAngle1 = ((float)rand() / (float)RAND_MAX) * 2.0f * PI;
            randomAngle2 = ((float)rand() / (float)RAND_MAX) * PI;
            randomRadius = ((float)rand() / (float)RAND_MAX) * GlobeRadius;
            h_P[i].x = randomRadius * cosf(randomAngle1) * sinf(randomAngle2);
            h_P[i].y = randomRadius * sinf(randomAngle1) * sinf(randomAngle2);
            h_P[i].z = randomRadius * cosf(randomAngle2);

            test = 1;
            int j;
            for(j = 0; j < i; j = j + 1)
            {
                dx = h_P[i].x - h_P[j].x;
                dy = h_P[i].y - h_P[j].y;
                dz = h_P[i].z - h_P[j].z;
                d = sqrtf(dx*dx + dy*dy + dz*dz);
                if(d < Diameter)
                {
                    test = 0;
                    break;
                }
            }
        }

        h_V[i].x = 0.0f;
        h_V[i].y = 0.0f;
        h_V[i].z = 0.0f;

        h_F[i].x = 0.0f;
        h_F[i].y = 0.0f;
        h_F[i].z = 0.0f;

        h_M[i] = 1.0f;
    }

    // Allocate device memory and copy initial arrays
    CUDA_CHECK(cudaMalloc((void**)&d_P, N * sizeof(float3)));
    CUDA_CHECK(cudaMalloc((void**)&d_V, N * sizeof(float3)));
    CUDA_CHECK(cudaMalloc((void**)&d_M, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_P, h_P, N * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, N * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_M, h_M, N * sizeof(float), cudaMemcpyHostToDevice));

    printf("\n To start timing type s.\n");
}

void nBody()
{
    // Host-side loop that launches a kernel for each time step.
    float time_local = 0.0f;
    float dt_local = DT;
    int drawCount = 0;
    int firstStepFlag = 1; // time == 0 case

    // Choose block/thread sizes. Single block with N threads (N < 1024).
    dim3 grid(1,1,1);
    dim3 block(N,1,1);

    while(time_local < RUN_TIME)
    {
        // Launch kernel to compute forces on each body and integrate one step.
        nbody_kernel<<<grid, block>>>(d_P, d_V, d_M, N, dt_local, Damp, firstStepFlag);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // If drawing intermediate steps, copy positions back every DRAW_RATE steps.
        if(drawCount == DRAW_RATE)
        {
            if(DrawFlag)
            {
                // copy positions back to host
                CUDA_CHECK(cudaMemcpy(h_P, d_P, N * sizeof(float3), cudaMemcpyDeviceToHost));
                drawPicture();
            }
            drawCount = 0;
        }

        // advance time and counters
        time_local = time_local + dt_local;
        drawCount = drawCount + 1;

        if(firstStepFlag == 1)
        {
            // unset first-step after first iteration
            firstStepFlag = 0;
        }
    }

    // final copy back for final draw
    CUDA_CHECK(cudaMemcpy(h_P, d_P, N * sizeof(float3), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_V, d_V, N * sizeof(float3), cudaMemcpyDeviceToHost));
}

int main(int argc, char** argv)
{
    if(argc < 3)
    {
        printf("\n You need to enter the number of bodies (an int)\n");
        printf(" and if you want to draw the bodies as they move (1 draw, 0 don't draw)\n");
        printf(" on the command line.\n");
        exit(0);
    }
    else
    {
        N = atoi(argv[1]);
        DrawFlag = atoi(argv[2]);
    }

    setup();

    int XWindowSize = 1000;
    int YWindowSize = 1000;

    glutInit(&argc,argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
    glutInitWindowSize(XWindowSize,YWindowSize);
    glutInitWindowPosition(0,0);
    glutCreateWindow("nBody GPU Test");

    GLfloat light_position[] = {1.0f, 1.0f, 1.0f, 0.0f};
    GLfloat light_ambient[]  = {0.0f, 0.0f, 0.0f, 1.0f};
    GLfloat light_diffuse[]  = {1.0f, 1.0f, 1.0f, 1.0f};
    GLfloat light_specular[] = {1.0f, 1.0f, 1.0f, 1.0f};
    GLfloat lmodel_ambient[] = {0.2f, 0.2f, 0.2f, 1.0f};
    GLfloat mat_specular[]   = {1.0f, 1.0f, 1.0f, 1.0f};
    GLfloat mat_shininess[]  = {10.0f};
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glShadeModel(GL_SMOOTH);
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
    glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
    glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_DEPTH_TEST);
    glutKeyboardFunc(keyPressed);
    glutDisplayFunc(drawPicture);

    float3 eye = {0.0f, 0.0f, 2.0f * GlobeRadius};
    float near = 0.2f;
    float far = 5.0f * GlobeRadius;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(-0.2f, 0.2f, -0.2f, 0.2f, near, far);
    glMatrixMode(GL_MODELVIEW);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    gluLookAt(eye.x, eye.y, eye.z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    glutMainLoop();

    // cleanup (will not be reached due to glutMainLoop, but kept for completeness)
    CUDA_CHECK(cudaFree(d_P));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_M));
    free(h_P);
    free(h_V);
    free(h_F);
    free(h_M);

    return 0;
}
