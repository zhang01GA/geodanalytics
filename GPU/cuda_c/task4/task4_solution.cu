#include <stdio.h>

#define N 2048 * 2048 // Number of elements in each vector

__global__ void saxpy(float scalar, float * x, float * y)
{
    // Determine our unique global thread ID, so we know which element to process
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if ( tid < N ) // Make sure we don't do more work than we have data!
        y[tid] = scalar * x[tid] + y[tid];
}

int main()
{
    float *x, *y;

    int size = N * sizeof (float); // The total number of bytes per vector

    cudaError_t ierrAsync;
    cudaError_t ierrSync;

    // Allocate memory
    cudaMallocManaged(&x, size);
    cudaMallocManaged(&y, size);

    // Initialize memory
    for( int i = 0; i < N; ++i )
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    int threads_per_block = 256;
    int number_of_blocks = (N / threads_per_block) + 1;

    saxpy <<< number_of_blocks, threads_per_block >>> ( 2.0f, x, y );

    ierrSync = cudaGetLastError();
    ierrAsync = cudaDeviceSynchronize(); // Wait for the GPU to finish
    if (ierrSync != cudaSuccess) { printf("Sync error: %s\n", cudaGetErrorString(ierrSync)); }
    if (ierrAsync != cudaSuccess) { printf("Async error: %s\n", cudaGetErrorString(ierrAsync)); }

    // Print out our Max Error
    float maxError = 0;
    for( int i = 0; i < N; ++i )
        if (abs(4-y[i]) > maxError) { maxError = abs(4-y[i]); }
    printf("Max Error: %.5f", maxError);

    // Free all our allocated memory
    cudaFree( x ); cudaFree( y );
}