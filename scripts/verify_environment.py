import sys
import platform
import subprocess
from pathlib import Path

print("=" * 60)
print("ENVIRONMENT & GPU VERIFICATION")
print("=" * 60)

# OS & Python
os_name = f"{platform.system()} {platform.release()} ({platform.version()})"
python_ver = sys.version.split()[0]
python_path = sys.executable

print(f"OS: {os_name}")
print(f"Python: {python_ver}")
print(f"Python Executable: {python_path}")

# PyTorch & CUDA
try:
    import torch
    torch_ver = torch.__version__
    cuda_avail = torch.cuda.is_available()
    cuda_ver = torch.version.cuda
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "None"
    gpu_count = torch.cuda.device_count()
    vram_bytes = torch.cuda.get_device_properties(0).total_memory if cuda_avail else 0
    vram_gb = vram_bytes / (1024**3)
    
    # Tensor allocation test
    if cuda_avail:
        x = torch.randn(100, 100, device='cuda')
        tensor_test = f"SUCCESS (Allocated tensor on {x.device})"
    else:
        tensor_test = "FAILED (CUDA not available)"
except Exception as e:
    torch_ver = f"Error: {e}"
    cuda_avail = False
    cuda_ver = "None"
    gpu_name = "None"
    vram_gb = 0.0
    tensor_test = f"Error: {e}"

print(f"PyTorch: {torch_ver}")
print(f"CUDA Available: {cuda_avail}")
print(f"PyTorch CUDA Version: {cuda_ver}")
print(f"GPU: {gpu_name}")
print(f"GPU VRAM: {vram_gb:.2f} GB")
print(f"CUDA Tensor Allocation: {tensor_test}")

# Ultralytics & CV packages
packages = {}
for pkg in ['ultralytics', 'cv2', 'pandas', 'numpy', 'scipy', 'sklearn', 'matplotlib', 'yaml', 'tqdm']:
    try:
        m = __import__(pkg)
        packages[pkg] = getattr(m, '__version__', 'Installed')
    except Exception as e:
        packages[pkg] = f"Error: {e}"

print("\nInstalled Packages:")
for k, v in packages.items():
    print(f"  {k:15s}: {v}")

print("=" * 60)
