import torch

print("Pytorch: ", torch.__version__)
print("GPU disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Tarjeta:", torch.cuda.get_device_name(0))