# xla_baseline Project Overview

## Introduction
The `xla_baseline` project is a **benchmarking and debugging tool for models utilizing PyTorch and the OpenXLA compiler**.
With this project, you can run PyTorch models in the XLA (PJRT) environment and export models to StableHLO (MLIR), an intermediate representation used for compiler optimization.

## Key Features
1. **Performance Benchmarking & Comparison**: Compares the execution results and benchmarks the speed between the native PyTorch backend (CPU/CUDA) and the PyTorch/XLA backend.
2. **StableHLO Bundle Export**: Traces the computation graph of PyTorch models to export StableHLO MLIR bytecode along with weight parameters as a bundle. The exported models can be loaded into an independent PJRT executor.
3. **Quantization Support**: To aid in compiler optimization, the project provides and exports not only base precision models (FP32/BF16) but also INT8 models utilizing Fake Quantization (QDQ) patterns and PTQ static quantization based on `torchao`.

## Directory Structure
* **`models/`**: Contains the definition blocks of reference models to be benchmarked (e.g., CNNs like ResNet/MobileNet, Transformers like ViT/BERT/GPT-2, and LLMs like Llama/DeepSeek).
* **`scripts/`**: A collection of Python scripts for model execution, performance comparison, and StableHLO export.
* **`utils/`**: Utilities for building PyTorch/XLA from source and analyzing XLA mapping results.
* **`results/`**: The output directory where StableHLO bundles, benchmark results, and CSV logs are saved upon script execution.
