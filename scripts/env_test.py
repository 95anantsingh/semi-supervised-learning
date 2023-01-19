try:
    import torch
except:
    print("\nPyTorch is not available.\n")
    exit()

if torch.cuda.is_available():
    print("\nCuda\t: Available")
    print("GPU(s)\t:",torch.cuda.device_count(),torch.cuda.get_device_name(),"\n")
else:
    print("\nCuda\t: Not Available\n")