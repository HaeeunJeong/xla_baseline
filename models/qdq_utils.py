import torch
import torch.nn as nn
from torch.ao.quantization.observer import MinMaxObserver, HistogramObserver

class QDQConv2d(nn.Module):
    def __init__(self, orig_conv: nn.Conv2d):
        super().__init__()
        self.orig_conv = orig_conv
        
        # qint8 for activations (symmetric) - using HistogramObserver to reduce outlier impact
        # per_tensor_symmetric: Only one scale parameter and one zero point for one tensor
        self.a_obs = HistogramObserver(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
        # qint8 for weights (symmetric)
        self.w_obs = MinMaxObserver(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
        
        # Execute weight overserver once to fix quantization parameters in this time
        self.w_obs(self.orig_conv.weight)
        w_scale, w_zp = self.w_obs.calculate_qparams()
        self.w_scale = w_scale.item()
        self.w_zp = int(w_zp.item())
        
        # Activation scale and zero point are temperary
        self.a_scale = 1.0
        self.a_zp = 0
        
        # Both activation and weights use signed int8
        self.quant_min_a = -128
        self.quant_max_a = 127
        self.quant_min_w = -128
        self.quant_max_w = 127
        self.calibrate = False
        
    # property: External codes can access the model.weight and model.bias
    # after this wrapper is applied
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
        
    # Do forward propagation to calibrate the quantization parameters,
    # reflecting the activation value range
    def forward(self, x):
        if self.calibrate:
            self.a_obs(x)
            a_scale, a_zp = self.a_obs.calculate_qparams()
            self.a_scale = a_scale.item()
            self.a_zp = int(a_zp.item())
            
        xq = torch.ops.quantized_decomposed.quantize_per_tensor(
            x, self.a_scale, self.a_zp, self.quant_min_a, self.quant_max_a, torch.int8)
        xdq = torch.ops.quantized_decomposed.dequantize_per_tensor(
            xq, self.a_scale, self.a_zp, self.quant_min_a, self.quant_max_a, torch.int8)
        
        wq = torch.ops.quantized_decomposed.quantize_per_tensor(
            self.orig_conv.weight, self.w_scale, self.w_zp, self.quant_min_w, self.quant_max_w, torch.int8)
        wdq = torch.ops.quantized_decomposed.dequantize_per_tensor(
            wq, self.w_scale, self.w_zp, self.quant_min_w, self.quant_max_w, torch.int8)
            
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
        
        self.quant_min_a = -128
        self.quant_max_a = 127
        self.quant_min_w = -128
        self.quant_max_w = 127
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
            x, self.a_scale, self.a_zp, self.quant_min_a, self.quant_max_a, torch.int8)
        xdq = torch.ops.quantized_decomposed.dequantize_per_tensor(
            xq, self.a_scale, self.a_zp, self.quant_min_a, self.quant_max_a, torch.int8)
        
        wq = torch.ops.quantized_decomposed.quantize_per_tensor(
            self.orig_linear.weight, self.w_scale, self.w_zp, self.quant_min_w, self.quant_max_w, torch.int8)
        wdq = torch.ops.quantized_decomposed.dequantize_per_tensor(
            wq, self.w_scale, self.w_zp, self.quant_min_w, self.quant_max_w, torch.int8)
            
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
    print("   [QDQ] Wrapping layers with QDQ modules...")
    model = apply_qdq(model)
    
    print("   [QDQ] Running robust calibration (100 batches)...")
    model.eval()
    for name, module in model.named_modules():
        if isinstance(module, (QDQConv2d, QDQLinear)):
            module.calibrate = True
            
    with torch.no_grad():
        for _ in range(100):
            # Using random noise with proper shape
            if isinstance(base_dummy_inputs, tuple):
                noise_inputs = tuple(torch.randn_like(t) for t in base_dummy_inputs)
                model(*noise_inputs)
            else:
                noise_inputs = torch.randn_like(base_dummy_inputs)
                model(noise_inputs)
                
    for name, module in model.named_modules():
        if isinstance(module, (QDQConv2d, QDQLinear)):
            module.calibrate = False
            
    return model
