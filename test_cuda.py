import torch
print("CUDA disponível:", torch.cuda.is_available())
print("GPU atual:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
