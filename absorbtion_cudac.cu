#include <curand_kernel.h>
#define N 216
#define T 256
#include <stdio.h>

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
                                int *result)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    int count_res = 0;

    /* Copy state to local memory for efficiency */
    curandStateXORWOW localState = state[id];
    /* Generate pseudo-random uniforms */
    for(int i = id; i < n; i+=N*T) {
        double mean_free_path = 1.0 / 0.5; // calculate mean free path
        double q = curand_uniform_double(&localState);
        double distance_to_collision = -log(q) * mean_free_path; // calculate distance to next collision
        double distance_to_surface = 1;
        if (distance_to_collision < distance_to_surface) {

                count_res++;}
}

        atomicAdd(result, count_res);
}

/* MAIN CODE */

int main(int argc,char* argv[])
{

int Q=1000000000; //declare initial number of particles 

for (int j=0; j<10; j++){

  curandStateXORWOW* state;

  /* allocate space on the GPU for the random states */
  cudaMalloc((void**) &state, N*T* sizeof(curandStateXORWOW));


  /* allocate an array of ints on the GPU */

  int *d_result;

  cudaMalloc((void**) &d_result, sizeof(int));
  cudaMemset(d_result, 0, sizeof(int));

  /* invoke the device kernel */

  setup_kernel<<<N, T>>>(state);

  generate_uniform_kernel<<<N, T>>>(state, Q, d_result);

  int result1;
  cudaMemcpy(&result1, d_result, sizeof(int), cudaMemcpyDeviceToHost);

  Q=result1;
       
 /* free the memory we allocated for the states and numbers */

cudaFree(state);
cudaFree(d_result);

}

  return 0;
}

