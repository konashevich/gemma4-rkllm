#!/bin/bash
set -e

cd /home/beast/gemma4-rkllm 
source rknn-llm/rkllm_env/bin/activate

echo "Downloading Gemma 4 model (16GB)..."
huggingface-cli download google/gemma-4-e4b-it --resume-download --local-dir ./gemma-4-e4b-it

echo "Model downloaded successfully."
echo "Converting Model to RKLLM Format..."
python3 convert_gemma4.py

echo "Process Complete! Your model is ready at gemma-4-e4b-it.rkllm"
