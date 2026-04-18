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
