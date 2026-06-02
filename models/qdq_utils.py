import torch
import torch.nn as nn
from torch.ao.quantization.observer import MinMaxObserver

class QDQConv2d(nn.Module):
    def __init__(self, orig_conv: nn.Conv2d):
        super().__init__()
        self.orig_conv = orig_conv
        
        self.a_obs = MinMaxObserver(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
        self.w_obs = MinMaxObserver(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
        
        # Initialize weight observer immediately since weights are static
        self.w_obs(self.orig_conv.weight)
        w_scale, w_zp = self.w_obs.calculate_qparams()
        self.w_scale = w_scale.item()
        self.w_zp = int(w_zp.item())
        
        self.a_scale = 1.0
        self.a_zp = 0
        
        self.quant_min = -128
        self.quant_max = 127
        self.calibrate = False
        
    @property
    def weight(self):
        return self.orig_conv.weight
        
    @property
    def bias(self):
        return self.orig_conv.bias

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'a_scale'] = torch.tensor(self.a_scale)
        destination[prefix + 'a_zp'] = torch.tensor(self.a_zp)
        destination[prefix + 'w_scale'] = torch.tensor(self.w_scale)
        destination[prefix + 'w_zp'] = torch.tensor(self.w_zp)
        
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if prefix + 'a_scale' in state_dict:
            self.a_scale = state_dict.pop(prefix + 'a_scale').item()
        if prefix + 'a_zp' in state_dict:
            self.a_zp = int(state_dict.pop(prefix + 'a_zp').item())
        if prefix + 'w_scale' in state_dict:
            self.w_scale = state_dict.pop(prefix + 'w_scale').item()
        if prefix + 'w_zp' in state_dict:
            self.w_zp = int(state_dict.pop(prefix + 'w_zp').item())
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        
    def forward(self, x):
        if self.calibrate:
            self.a_obs(x)
            a_scale, a_zp = self.a_obs.calculate_qparams()
            self.a_scale = a_scale.item()
            self.a_zp = int(a_zp.item())
            
        xq = torch.ops.quantized_decomposed.quantize_per_tensor(
            x, self.a_scale, self.a_zp, self.quant_min, self.quant_max, torch.int8)
        xdq = torch.ops.quantized_decomposed.dequantize_per_tensor(
            xq, self.a_scale, self.a_zp, self.quant_min, self.quant_max, torch.int8)
        
        wq = torch.ops.quantized_decomposed.quantize_per_tensor(
            self.orig_conv.weight, self.w_scale, self.w_zp, self.quant_min, self.quant_max, torch.int8)
        wdq = torch.ops.quantized_decomposed.dequantize_per_tensor(
            wq, self.w_scale, self.w_zp, self.quant_min, self.quant_max, torch.int8)
            
        return nn.functional.conv2d(xdq, wdq, self.orig_conv.bias, 
                                    self.orig_conv.stride, self.orig_conv.padding, 
                                    self.orig_conv.dilation, self.orig_conv.groups)

class QDQLinear(nn.Module):
    def __init__(self, orig_linear: nn.Linear):
        super().__init__()
        self.orig_linear = orig_linear
        
        self.a_obs = MinMaxObserver(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
        self.w_obs = MinMaxObserver(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
        
        self.w_obs(self.orig_linear.weight)
        w_scale, w_zp = self.w_obs.calculate_qparams()
        self.w_scale = w_scale.item()
        self.w_zp = int(w_zp.item())
        
        self.a_scale = 1.0
        self.a_zp = 0
        
        self.quant_min = -128
        self.quant_max = 127
        self.calibrate = False
        
    @property
    def weight(self):
        return self.orig_linear.weight
        
    @property
    def bias(self):
        return self.orig_linear.bias

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'a_scale'] = torch.tensor(self.a_scale)
        destination[prefix + 'a_zp'] = torch.tensor(self.a_zp)
        destination[prefix + 'w_scale'] = torch.tensor(self.w_scale)
        destination[prefix + 'w_zp'] = torch.tensor(self.w_zp)
        
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if prefix + 'a_scale' in state_dict:
            self.a_scale = state_dict.pop(prefix + 'a_scale').item()
        if prefix + 'a_zp' in state_dict:
            self.a_zp = int(state_dict.pop(prefix + 'a_zp').item())
        if prefix + 'w_scale' in state_dict:
            self.w_scale = state_dict.pop(prefix + 'w_scale').item()
        if prefix + 'w_zp' in state_dict:
            self.w_zp = int(state_dict.pop(prefix + 'w_zp').item())
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        
    def forward(self, x):
        if self.calibrate:
            self.a_obs(x)
            a_scale, a_zp = self.a_obs.calculate_qparams()
            self.a_scale = a_scale.item()
            self.a_zp = int(a_zp.item())
            
        xq = torch.ops.quantized_decomposed.quantize_per_tensor(
            x, self.a_scale, self.a_zp, self.quant_min, self.quant_max, torch.int8)
        xdq = torch.ops.quantized_decomposed.dequantize_per_tensor(
            xq, self.a_scale, self.a_zp, self.quant_min, self.quant_max, torch.int8)
        
        wq = torch.ops.quantized_decomposed.quantize_per_tensor(
            self.orig_linear.weight, self.w_scale, self.w_zp, self.quant_min, self.quant_max, torch.int8)
        wdq = torch.ops.quantized_decomposed.dequantize_per_tensor(
            wq, self.w_scale, self.w_zp, self.quant_min, self.quant_max, torch.int8)
            
        return nn.functional.linear(xdq, wdq, self.orig_linear.bias)

def apply_qdq(module: nn.Module) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(module, name, QDQConv2d(child))
        elif isinstance(child, nn.Linear):
            setattr(module, name, QDQLinear(child))
        else:
            apply_qdq(child)
    return module

def apply_qdq_and_calibrate(model: nn.Module, base_dummy_inputs) -> nn.Module:
    """
    Applies QDQ blocks and runs a 100-loop calibration with variance-scaled noise
    to simulate a realistic dataset and calculate robust min/max bounds.
    """
    model = apply_qdq(model)
    
    # Enable calibration mode
    for m in model.modules():
        if hasattr(m, 'calibrate'):
            m.calibrate = True
    
    # Extract the shape from dummy inputs to generate varied batches
    if isinstance(base_dummy_inputs, dict):
        first_key = list(base_dummy_inputs.keys())[0]
        base_shape = base_dummy_inputs[first_key].shape
    elif isinstance(base_dummy_inputs, tuple):
        base_shape = base_dummy_inputs[0].shape
    else:
        base_shape = base_dummy_inputs.shape

    with torch.no_grad():
        print("   [QDQ] Running robust calibration (100 batches)...")
        for i in range(100):
            # Simulate dataset with varying magnitude/variance
            scale_factor = 1.0 + (i / 100.0) * 2.0  # Scales from 1.0 to 3.0
            noise = torch.randn(base_shape) * scale_factor
            
            if isinstance(base_dummy_inputs, dict):
                inputs = {k: noise for k in base_dummy_inputs}
                model(**inputs)
            elif isinstance(base_dummy_inputs, tuple):
                inputs = (noise,) * len(base_dummy_inputs)
                model(*inputs)
            else:
                model(noise)
            
    # Disable calibration mode
    for m in model.modules():
        if hasattr(m, 'calibrate'):
            m.calibrate = False
            
    return model
