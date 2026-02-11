import torch

print("torch =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("cuda available =", torch.cuda.is_available())
print("cuda device count =", torch.cuda.device_count())

if torch.cuda.is_available():
    print("cuda name =", torch.cuda.get_device_name(0))
    x = torch.randn(3, 3, device="cuda")
    print("tensor on =", x.device)
