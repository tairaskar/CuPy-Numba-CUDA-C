import numpy as np
from numba import cuda, float64
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64
from numba import jit
import numba

@cuda.jit
def gpu_random_numbers(rng_states, out, N):
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    stride = cuda.blockDim.x * cuda.gridDim.x
    for i in range(tid, N, stride):
        out[i] = cuda.random.xoroshiro128p_uniform_float64(rng_states, tid)


num_numbers = 100000000
block_size = 256
grid_size = 216

rng_states = create_xoroshiro128p_states(grid_size * block_size, seed=1234)
d_out = cuda.device_array(num_numbers, dtype=np.float64)
gpu_random_numbers[grid_size, block_size](rng_states, d_out, num_numbers)
     

