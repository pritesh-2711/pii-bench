import sys
print("Python:", sys.version)

pkgs = ["torch", "transformers", "seqeval", "datasets", "span_marker", "accelerate"]
for pkg in pkgs:
    try:
        m = __import__(pkg)
        ver = getattr(m, "__version__", "installed")
        print(f"  {pkg}: {ver}")
    except ImportError:
        print(f"  {pkg}: NOT installed")

try:
    import torch
    print("\nCUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
except Exception as e:
    print("torch check failed:", e)
