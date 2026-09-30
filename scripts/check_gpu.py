
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:", torch.cuda.get_device_name(i))
        props = torch.cuda.get_device_properties(i)
        print(f"  VRAM: {props.total_memory / 1024**3:.2f} GB")
else:
    print("Running on CPU.")
