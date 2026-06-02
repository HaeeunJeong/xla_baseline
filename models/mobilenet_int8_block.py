from __future__ import annotations
import torch
from .mobilenet_block import get_model as base_get_model, get_dummy_input
from .qdq_utils import apply_qdq_and_calibrate
from scripts.export_shlo import make_inputs

def get_model():
    model = base_get_model()
    dummy_input = get_dummy_input()
    inputs = make_inputs(dummy_input, "mobilenet")
    model = apply_qdq_and_calibrate(model, inputs)
    return model
