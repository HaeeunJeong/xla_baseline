# Model Quantization Principles and Export Guide

This project supports **Post-Training Static Quantization (PTQ)** to help the XLA compiler easily detect quantization points in the model and optimize them with hardware instructions (like INT8 GEMM).

## Quantization Principles (Fake Quantization)

This project inserts **QDQ (Quantize-Dequantize)** patterns to simulate the INT8 range without losing the floating-point calculation graph. The core logic is implemented in `models/qdq_utils.py`.

1. **Calibration via Observers**:
   - To calculate the quantization parameters (`scale` and `zero_point`), the value distribution before and after passing through the layer is observed.
   - **Weight**: Since weights are fixed, a `MinMaxObserver` is used to calculate the scale based on the minimum and maximum ranges.
   - **Activation**: Values change depending on the input data and may contain outliers, so a `HistogramObserver` is used to determine the optimal clipping range.
   - Both observers operate using symmetric INT8 quantization (`qint8` and `per_tensor_symmetric`) bounded between `-128` and `127`.

2. **Quantization Simulation Process (`forward`)**:
   - `quantize_per_tensor`: Clips the values to INT8 resolution.
   - `dequantize_per_tensor`: Immediately restores the INT8 values back to FP32/BF16 data types.
   - Through this process, the data type remains floating-point (keeping the computation graph connected), but the precision is degraded to INT8, creating a "Fake Quantization" state.

3. **Automatic Module Replacement**:
   - When the `apply_qdq_and_calibrate()` function is called, existing `nn.Conv2d` and `nn.Linear` layers in the model are replaced entirely with `QDQConv2d` and `QDQLinear` modules applying the above logic. It then automatically calibrates the optimal ranges by passing random dummy inputs for 100 iterations.

## How to Export Quantized Models

To export PTQ-applied INT8 models in StableHLO format, use the `scripts/export_ptq_shlo.py` script instead of the default export script.

```bash
conda activate torch-xla

# Export quantization for a specific model (e.g., resnet)
python scripts/export_ptq_shlo.py resnet

# Export quantization for all models (automatically discovered when omitted)
python scripts/export_ptq_shlo.py
```

### Execution Flow
1. It prioritizes models with `_int8` and `_bf16` suffixes in the `models/` directory.
2. It injects the XNNPACK Quantizer (or the QDQ pattern from `qdq_utils.py`) into the model and performs calibration with dummy inputs to fix the scale and zero-point.
3. The static quantized graph (FX Graph) is processed through the `torch_xla` toolchain and saved in StableHLO format (`MLIR` + `calibrated_pytorch_model.pt`) into the `results/xla/StableHLO/` directory.
