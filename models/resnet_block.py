from __future__ import annotations
import torch
from torchvision.models import resnet18, ResNet18_Weights as W
from PIL import Image
import torchvision.transforms as transforms
import os

def get_model() -> torch.nn.Module:
    return resnet18(weights=W.IMAGENET1K_V1).eval()

def get_dummy_input() -> tuple[torch.Tensor]:
    img_path = os.path.join(os.path.dirname(__file__), "../inputs/cat.jpeg")
    img = Image.open(img_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = preprocess(img).unsqueeze(0)
    return (tensor,)
