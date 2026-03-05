# xla_baseline

*OpenXLA compiler benchmarks & debugging tools*

This repository is built to export and test StableHLO modules:
- **PyTorch baseline** execution (CPU/CUDA)
- **Exporting** models to **StableHLO** MLIR bundles
- **PyTorch/XLA (PJRT)** execution

---
## 📂 Repository Layout

```text
xla_baseline/
├─ env/           # Conda environment specifications
├─ models/        # Reference models for benchmarking (CNN, Transformer, and LLM.)
├─ scripts/       # Entry-point Python scripts to execute tests, tracing, and exports
└─ utils/         # Installation scripts, mapping analyzers, and other utilities
```
---

## 🛠 Prerequisites

- **Python**: 3.10+ recommended (The `env/torch-xla` environment targets Python 3.13)
- **PyTorch / torch-xla**: The major and minor versions of PyTorch must exactly match the `torch-xla` and `torchvision` versions for successful execution.
  - (e.g., PyTorch: **2.9.0**, torch-xla: **2.9.8-rc3**, torchvision: **0.24.0**)


---

## 🚀 Getting Started (Installation)

First, clone the repository:
```bash
git clone https://github.com/HaeeunJeong/xla_baseline.git
cd xla_baseline
```

Depending on your target execution device (CPU vs. GPU), the environment setup differs significantly.

### CPU Execution & StableHLO Export (Build PyTorch/XLA from Source via Conda)

If you just want to run the models on a **CPU** or your main goal is to use the **StableHLO export functionality**, building PyTorch/XLA from source within a Conda environment is the way to go.

You can use the provided `utils/set_torch_xla.sh` script to check out specific commits of the PyTorch, PyTorch/XLA, and torchvision repositories and build them from source.

Run the script from the `utils/` folder:
```bash
cd utils
bash set_torch_xla.sh
```
> **⚠️ Memory Requirement Warning:** Compiling `torch-xla` from source **requires up to ~170GB of memory**. On standard workstations, the compilation can easily crash the system due to Out-Of-Memory (OOM) errors. The `set_torch_xla.sh` script natively allocates a temporary 64GB swap file (`swapfile2`) on your root disk during the build process. Please ensure you have sufficient disk space.

### GPU Execution (Using Official Prebuilt Docker Images)

If you want to run the models on a **GPU (CUDA)**, using the official prebuilt Docker images provided by the PyTorch/XLA team is **highly recommended (and often the only viable option)**.

> **Why can't I build with CUDA locally?** The `torch-xla` package officially supported CUDA (up to version 12.4) only out to version 2.5.0. Since then, the native CUDA support was entirely removed from their standard build process for newer versions. Therefore, using their verified Docker container is the best and safest path for GPU execution.

Check the available tags using the `gcloud` CLI tool:
```bash
gcloud artifacts docker tags list us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla
```

An example of running a recent nightly image (`nightly_3.11_cuda_12.8_20250407`):
```bash
docker run \
--net=host \
--gpus all \
--shm-size=16g \
--name torch-xla-prebuilt \
-itv {host_system_dir}:{container_dir} \
-d us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.11_cuda_12.8_20250407 /bin/bash

docker exec -it torch-xla-prebuilt /bin/bash
```

Once inside the container, install the required CUDA Toolkit (e.g., 12.8):
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8

pip uninstall torch
pip install torch
```

---

## 🏃 Usage Guide: Scripts & Utilities

As mentioned, all scripts in the `scripts/` directory **require the `torch-xla` environment to be activated.** If you encounter module loading errors, ensure the library paths are correctly appended in your shell or `~/.bashrc`.

> **Common PJRT Environment Variables:**
> Force the execution device using `PJRT_DEVICE=CPU` (or `PJRT_DEVICE=CUDA` if running inside the Docker container).

### 1) Baseline vs XLA Execution

* **`pytorch_baseline.py`**
  Benchmarks the baseline performance using the native PyTorch backend (CPU/CUDA) without XLA.
  ```bash
  python scripts/pytorch_baseline.py --device cpu <MODEL>
  ```
  If `<MODEL>` is not specified, it tests all available models.

* **`compare_xla_torch.py`**
  Directly compares the execution behavior of the default PyTorch backend versus the `torch_xla` backend side-by-side.
  ```bash
  PJRT_DEVICE=CPU python scripts/compare_xla_torch.py <MODEL>
  ```

* **`compile_run_xla.py`**
  Dynamically traces a `torch.nn.Module`, compiles the lowered graph into HLO, and executes it via the XLA compiler/runtime.
  ```bash
  PJRT_DEVICE=CPU python scripts/compile_run_xla.py <MODEL>
  ```

### 2) StableHLO Exporting & Execution

* **`export_shlo.py`**
  Exports a traced PyTorch module to a **StableHLO bundle** (containing both the MLIR bytecode and the parameter weight distributions). Look for the saved artifacts in `results/xla/StableHLO/`.
  ```bash
  python scripts/export_shlo.py <MODEL>
  ```

* **`export_debug.py`**
  A robust wrapper around the regular export pipeline that automatically injects `write_mlir_debuginfo` calls into the extracted FX graph. This attaches `fx_id` tag bounds straight into the generated StableHLO MLIR, improving trace analysis.
  ```bash
  python scripts/export_debug.py --xla_debuginfo <MODEL>
  ```

* **`shlo_compile_run.py`**
  Reads the previously extracted **StableHLO bundle** on disk and directly executes the MLIR bytecode natively via PJRT.
  ```bash
  PJRT_DEVICE=CPU python scripts/shlo_compile_run.py \
    --bundle ./results/xla/StableHLO/<MODEL>_stablehlo \
    --batch 32
  ```

### 3) Utilities (`utils/`)

* **`extract_table.py`**: Reads `export_debug.py` tagged MLIR forward functions, computes operational boundaries, and extracts an ATen-to-StableHLO backend mapping table as JSON.
* **`compare_mappings.py`**: Compares various ATen-to-StableHLO lowering schema variations across multiple mapping tables across models.

---

## 🧩 Supported Models

The `models/` directory contains reference model implementations in the format of `<MODEL>_block.py`. The scripts dynamically route to these definitions based on the provided string argument.

| Category | Key (`<MODEL>`) | Source/Library | Notes |
| --- | --- | --- | --- |
| Custom | `mm` | `models/mm_block.py` | Simple tensor matrix multiplication (Matmul) |
| Custom | `conv` | `models/conv_block.py` | Basic Conv-Flatten-Linear-ReLU sequence |
| **CNN** | `resnet` | `torchvision` (ResNet-18) | Classic ImageNet pre-trained backbone |
| **CNN** | `mobilenet` | `torchvision` (MobileNet v3 S) | Lightweight, mobile-oriented CNN |
| **Transformer** | `vit` | `torchvision` (ViT-B/16) | Vision Transformer baseline block |
| **Transformer** | `bert` | HuggingFace (`bert-base-uncased`) | Token-level text encoder |
| **Transformer** | `gpt2` | HuggingFace | Standard Decoder architecture |
| **LLM** | `llama` | HuggingFace (`meta-llama/Llama-3.2-1B`) | Meta's lightweight generation LLM (Base version) |
| **LLM** | `deepseek` | HuggingFace (`DeepSeek-R1-Distill-Qwen-1.5B`) | Reasoning-focused distilled LLM |

> **HuggingFace Authorization Note:** Downloading weights for models like Llama or DeepSeek typically requires accepting terms and configuring a user token. It is recommended to log in via CLI:
> ```bash
> pip install -U "huggingface_hub[cli]"
> huggingface-cli login
> ```

---

## 🛠 Tips: XLA Debugging & Dumps

1. **Dump Complete IR/HLO Text**
   This outputs the unoptimized/optimized HLO graph artifacts produced during compilation.
   ```bash
   export XLA_FLAGS="--xla_dump_to=./out"
   ```

2. **Metrics & Profiling Summaries**
   Enable this debug level to see detailed metrics about compilation times, execution counts, tensor transfers between host/device, and pinpoint ops that failed to lower to XLA.
   ```bash
   export PT_XLA_DEBUG_LEVEL=2
   ```

3. **Applying Multiple XLA Environment Flags**
   You can chain multiple backend configs together, for example, enabling latency hiding graph scheduling:
   ```bash
   export XLA_FLAGS="${XLA_FLAGS} --xla_dump_hlo_as_text --xla_gpu_enable_latency_hiding_scheduler=true"
   ```
