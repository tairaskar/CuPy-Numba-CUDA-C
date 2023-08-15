import cupy as cp
from numba import cuda

# Allocate an array on the GPU
array_size = 200000000
a_gpu = cp.empty(array_size, dtype=cp.float64)

# Create a new generator with a specific seed
gen = cp.random.Generator(cp.random.XORWOW(1234))

# Generate 10 random numbers between 0 and 1
a_gpu = gen.random(array_size, dtype=cp.float64)




