
import torch
import torchaudio
import torchvision
import transformers
import peft
import trl
import datasets
import wandb    
import lighteval

print("transformers.__version__:", transformers.__version__)
print("peft.__version__", peft.__version__)
print("trl.__version__:", trl.__version__)
print("datasets.__version__:", datasets.__version__)
print("wandb.__version__:", wandb.__version__)
print("lighteval.__version__:", lighteval.__version__)


def basic_calculation(a: int, b: int) -> torch.Tensor:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    tensor_a = torch.arange(start=1, end=a + 1).unsqueeze(0)
    tensor_b = torch.arange(start=1, end=b + 1).unsqueeze(0)
    tensor_product = torch.matmul(tensor_a.T, tensor_b).to(device)
    return tensor_product


print("torch.__version__:", torch.__version__)
print("torchvision.__version__:", torchvision.__version__)
print("torchaudio.__version__:", torchaudio.__version__)
print("torch.cuda.is_available:", torch.cuda.is_available())
print()
for i in range(torch.cuda.device_count()):
    print("Device", i, ": ", end="")
    print(torch.cuda.get_device_properties(i))
print()
print("Test calculation")
print(basic_calculation(2, 3))