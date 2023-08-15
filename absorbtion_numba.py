import numpy as np
from numba import cuda, float64
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64
from numba import jit
from math import log
import numba as nb

@cuda.jit
def gpu_random_numbers(rng_states, N, arr):
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    stride = cuda.blockDim.x * cuda.gridDim.x
    count_res = 0
    for i in range(tid, N, stride):
        mean_free_path = nb.float64(1.0/0.5)
        q = cuda.random.xoroshiro128p_uniform_float64(rng_states, tid)
        distance_to_collision =nb.float64( -log(q) * mean_free_path)
        distance_to_surface = nb.float64(1.0)
        if distance_to_collision < distance_to_surface:
                count_res +=1
    cuda.atomic.add(arr, 0, count_res)


num_numbers = 1000000000

block_size = 256
grid_size = 216

for i in range(10):
	
	rng_states = create_xoroshiro128p_states(grid_size * block_size, seed=1234)
	
	arr = np.zeros(1, dtype=np.int32)
	
	gpu_random_numbers[grid_size, block_size](rng_states, num_numbers, arr)
	
	result = arr[0]
	
	num_numbers = num_numbers - result
 
