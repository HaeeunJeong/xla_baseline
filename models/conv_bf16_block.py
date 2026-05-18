from __future__ import annotations
import torch
from .conv_block import get_model as base_get_model, get_dummy_input

def get_model():
    model = base_get_model()
    return model.to(torch.bfloat16)
