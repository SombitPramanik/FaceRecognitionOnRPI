import multiprocessing
import os

# Function to calculate square
def calculate_square(number):
    square = number ** 2
    process_id = os.getpid()
    print(f"Process {process_id}: Square of {number} is {square}")

if __name__ == "__main__":
    try:
        use_gpu = False
        try:
            # Check if CUDA capable GPU is available
            import torch

            if torch.cuda.is_available():
                use_gpu = True
                print("GPU is available!")
            else:
                print("CUDA information Invalid or Not Available")
        except ImportError:
            print("torch is not installed. Falling back to CPU.")

        pool = multiprocessing.Pool(processes=(multiprocessing.cpu_count()-2))

        # Distribute work among processes
        numbers = range(1, 11)  # Example numbers to square
        pool.map(calculate_square, numbers)

        pool.close()
        pool.join()

    except ValueError as ve:
        print(f"Error: {ve}")
