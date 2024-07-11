import numpy as np
from numba import jit
import time

# Define a function to calculate squares using GPU
@jit
def calculate_square_gpu(arr):
    return arr ** 2

def main():
    # Generate a large numpy array
    array_size = 1000000000
    arr = np.arange(array_size).astype(np.float32)

    # Time CPU calculation
    start_time = time.time()
    result_cpu = arr ** 2
    cpu_time = time.time() - start_time

    print(f"Time taken for CPU: {cpu_time:.4f} seconds")

    # Time GPU calculation
    start_time = time.time()
    result_gpu = calculate_square_gpu(arr)
    gpu_time = time.time() - start_time

    print(f"Time taken for GPU: {gpu_time:.4f} seconds")

    # Compare results (just to verify correctness)
    np.testing.assert_allclose(result_cpu, result_gpu, rtol=1e-5)

if __name__ == "__main__":
    main()
