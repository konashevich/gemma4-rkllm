# Plan for Converting Google Gemma 4 for Rockchip RK3588 NPU

This document outlines the step-by-step plan for converting the newly released **Google Gemma 4** Large Language Model (specifically the edge-optimized **E2B** or **E4B** variants) into the `.rkllm` format required for the RK3588's Neural Processing Unit (NPU).

## 1. Prerequisites

### Conversion Environment (Host PC)
Converting Large Language Models requires significant RAM and standard GPU resources. The conversion **cannot** be done directly on the CM3588.
- **OS:** Ubuntu 20.04 or 22.04 (Native or WSL2)
- **RAM:** Minimum 16GB (32GB+ recommended)
- **Software:** 
  - Python 3.8
  - Python venv

### Target Environment (CM3588)
- Up-to-date RK3588 kernel and NPU driver (v0.9.8+ recommended for LLMs).
- `rkllm-runtime` library installed.

---

## 2. Setting Up the Conversion Environment (Host PC)

1. Clone the Rockchip LLM Toolkit repository:
   ```bash
   git clone https://github.com/airockchip/rknn-llm.git
   cd rknn-llm
   git reset 8623edd0559a07e7127876d685f2b7ca8b83590c --hard
   ```

2. Create a Python virtual environment and install dependencies:
   ```bash
   python3.8 -m venv rkllm_env
   source rkllm_env/bin/activate
   pip install -U pip
   ```

3. Install the RKLLM Toolkit wheel:
   ```bash
   pip install rkllm-toolkit/packages/rkllm_toolkit-1.1.4-cp38-cp38-linux_x86_64.whl
   ```

---

## 3. Acquiring the Gemma 4 Model

The RKLLM toolkit directly ingests Hugging Face model structures.

1. Install `huggingface-cli` and download the Gemma 4 E4B Instruct model (or E2B for lighter memory footprint):
   ```bash
   huggingface-cli download google/gemma-4-e4b-it --local-dir ./gemma-4-e4b-it
   ```

*Note: You will need to accept Google's terms on the Hugging Face website and log in using `huggingface-cli login` to download official restricted models.*

---

## 4. Converting Gemma 4 to RKLLM (with Quantization)

Rockchip provides Python APIs to compile and quantize models. You will create a Python script on your Host PC to perform the conversion. 4-bit weight quantization (`w4a16` or `w8a8`) is standard for edge deployment like the RK3588.

1. Create a script named `convert_gemma4.py`:

```python
from rkllm.api import RKLLM

# 1. Initialize RKLLM
rkllm = RKLLM()

# 2. Load the Hugging Face Model
print("Loading Gemma 4 model...")
ret = rkllm.load_huggingface(
    model='./gemma-4-e4b-it',
    model_name='gemma' # Note: Check latest rkllm docs if 'gemma4' is needed as the model_name flag
)
if ret != 0:
    print("Load model failed!")
    exit(ret)

# 3. Build with 4-bit Quantization
print("Building RKLLM model...")
ret = rkllm.build(
    do_quantization=True,
    optimization_level=1,
    quantized_dtype='w4a16', # Optimal for RK3588 memory
    target_platform='rk3588'
)
if ret != 0:
    print("Build model failed!")
    exit(ret)

# 4. Export the final .rkllm file
print("Exporting RKLLM model...")
ret = rkllm.export_rkllm('./gemma-4-e4b-it.rkllm')
if ret != 0:
    print("Export model failed!")
    exit(ret)

print("Conversion Successful!")
```

2. Run the script. This will output the NPU-ready `gemma-4-e4b-it.rkllm` file.

---

## 5. Deployment on the CM3588 NPU

1. **Transfer the Model:** Use SCP or a USB drive to copy `gemma-4-e4b-it.rkllm` to the CM3588.
2. **Install Runtime:** Ensure the C++ `librkllmrt.so` is appropriately installed on the CM3588.
3. **Execute:** Use Rockchip's provided `rkllm_server` or the C++ prompt demo to interact with your Gemma 4 model locally utilizing the NPU.

## Alternative: rk-llama.cpp
An active alternative is the open-source `rk-llama.cpp`. Since Gemma 4 is a recent release (April 2026), `rk-llama.cpp` forks are typically faster at merging support for new model architectures than the official RKLLM toolkit. If `rknn-llm` fails to parse the Gemma 4 blocks, try converting the model to GGUF format and running it via `rk-llama.cpp` with NPU offloading enabled.
