#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#define blockCount 216
#define threadsPerBlock 256
#define sampleCount 1000000000

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

__global__ void setup_kernel(curandStateXORWOW *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(1234, id, 0, &state[id]);
}

__global__ void generate_uniform_kernel(curandStateXORWOW *state,
                                int n,
                                float *result)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    /* Copy state to local memory for efficiency */
    curandStateXORWOW localState = state[id];
    /* Generate pseudo-random uniforms */
    for(int i = id; i < n; i+=threadsPerBlock*blockCount) {
        result[id] = curand_uniform(&localState);
    }
}

int main(int argc, char *argv[])
{

    int totalThreads = threadsPerBlock * blockCount;

    curandStateXORWOW *devStates;

    float *devResults;


    /* Allocate space for results on device */
    CUDA_CALL(cudaMalloc((void **)&devResults, sampleCount *
              sizeof(float)));

    CUDA_CALL(cudaMalloc((void **)&devStates, totalThreads *
                  sizeof(curandState)));

    setup_kernel<<<blockCount, threadsPerBlock>>>(devStates);


    generate_uniform_kernel<<<blockCount, threadsPerBlock>>>(devStates, sampleCount, devResults);


    CUDA_CALL(cudaFree(devStates));

    CUDA_CALL(cudaFree(devResults));

    
    return EXIT_SUCCESS;
}

