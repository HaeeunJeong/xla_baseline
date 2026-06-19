# xla_baseline Usage Guide

All scripts must be executed within the `torch-xla` environment. Please activate the virtual environment before running them.

```bash
conda activate torch-xla
```

## Main Scripts Usage

Scripts in the `scripts/` directory generally take a `<MODEL>` argument. If the `<MODEL>` argument is omitted, the scripts will automatically and sequentially execute all models defined with names ending in `_block` within the `models/` directory.

### 1. Execution & Comparison

- **PyTorch Native Backend Benchmark** (`pytorch_baseline.py`)
  ```bash
  python scripts/pytorch_baseline.py --device cpu <MODEL>
  ```
- **XLA vs PyTorch Comparison** (`compare_xla_torch.py`)
  ```bash
  PJRT_DEVICE=CPU python scripts/compare_xla_torch.py <MODEL>
  ```
- **XLA Model Compilation & Immediate Execution** (`compile_run_xla.py`)
  ```bash
  PJRT_DEVICE=CPU python scripts/compile_run_xla.py <MODEL>
  ```

### 2. StableHLO Model Export

Export the model in StableHLO format for analysis or to be used in a standalone runtime.
The artifacts are saved in `results/xla/StableHLO/<MODEL>_stablehlo/`.

- **Base Model StableHLO Export** (`export_shlo.py`)
  ```bash
  python scripts/export_shlo.py <MODEL>
  ```
- **PTQ (Static Quantization) Model StableHLO Export** (`export_ptq_shlo.py`)
  Exports low-precision/quantized models like INT8 or BF16.
  ```bash
  python scripts/export_ptq_shlo.py <MODEL>
  ```
- **Direct Execution of Exported StableHLO** (`shlo_compile_run.py`)
  ```bash
  PJRT_DEVICE=CPU python scripts/shlo_compile_run.py --bundle ./results/xla/StableHLO/<MODEL>_stablehlo
  ```

## Environment Variables
* `PJRT_DEVICE=CPU` (or `CUDA`): Forces the execution device for XLA.
* `PT_XLA_DEBUG_LEVEL=2`: Outputs compilation times and additional detailed debugging logs.
