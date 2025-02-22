import torch

def main():
    print("Hello, world!")

    # Check if CUDA is available
    if torch.cuda.is_available():
        # Get the name of the first GPU device
        gpu_name = torch.cuda.get_device_name(0)
        print("CUDA is available. GPU detected:", gpu_name)
        
        # Verify that the GPU is an Nvidia T4
        if "T4" in gpu_name:
            print("Confirmed: Running on an Nvidia T4 GPU.")
        else:
            print("Warning: Expected an Nvidia T4 GPU but got:", gpu_name)
        
        # Smoke test: perform a simple tensor addition on the GPU
        a = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        b = torch.tensor([4.0, 5.0, 6.0], device="cuda")
        c = a + b
        print("Tensor addition result:", c)
    else:
        print("CUDA is not available. Please check your GPU configuration.")

if __name__ == "__main__":
    main()
