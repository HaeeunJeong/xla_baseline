from __future__ import annotations
import torch
from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
from .vit_block import get_model as base_get_model, get_dummy_input

def get_model():
    model = base_get_model()
    quantize_(model, Int8DynamicActivationInt8WeightConfig())
    return model
