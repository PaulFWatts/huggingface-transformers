import torch

print(torch.version.cuda)

if torch.cuda.is_available():
        print("CUDA is available. GPU is ready to use.")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory}")
