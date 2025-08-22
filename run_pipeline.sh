#!/bin/bash

# This script automates the entire pipeline:
# 1. Checks for and downloads the required LLM model if it's missing.
# 2. Runs the transcription script (ref_x_max.py).
# 3. Runs the vLLM analysis script (ref_x_vllm.py).

# --- Configuration ---
MODEL_DIR="model"
MODEL_PATH="$MODEL_DIR/LLM.xyz"
DOWNLOAD_URL="https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf"
TEMP_MODEL_NAME="$MODEL_DIR/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf"

# --- 1. Model Check and Download ---
# Check if the model file already exists.
if [ ! -f "$MODEL_PATH" ]; then
    echo "Model file not found at $MODEL_PATH."
    echo "Starting download... (This may take a while depending on your network speed)"

    # Create the model directory if it doesn't exist
    mkdir -p "$MODEL_DIR"

    # Download the model using wget
    wget -O "$TEMP_MODEL_NAME" "$DOWNLOAD_URL"

    # Check if the download was successful by checking the exit code ($?)
    if [ $? -eq 0 ]; then
        echo "Download complete."
        # Rename the downloaded file to the expected name
        mv "$TEMP_MODEL_NAME" "$MODEL_PATH"
        echo "Model successfully set up at $MODEL_PATH."
    else
        echo "ERROR: Model download failed. Please check the URL or your network connection."
        # Exit the script with an error code to prevent further execution
        exit 1
    fi
else
    echo "Model file already exists. Skipping download."
fi

echo "--------------------------------------------------"

# --- 2. Run Transcription Script ---
echo "Starting the transcription process (ref_x_max.py)..."
python ref_x_max.py

# Check if the transcription script ran successfully
if [ $? -ne 0 ]; then
    echo "ERROR: The transcription script (ref_x_max.py) failed. Aborting pipeline."
    exit 1
fi

echo "Transcription process finished."
echo "--------------------------------------------------"

# --- 3. Run vLLM Script ---
echo "Starting the vLLM process (ref_x_vllm.py)..."
python ref_x_vllm.py \
    --input output_multi8/final_results_lasthour.json \
    --output output_vllm/ \
    --model "$MODEL_PATH"

if [ $? -ne 0 ]; then
    echo "ERROR: The vLLM script (ref_x_vllm.py) failed."
    exit 1
fi

echo "vLLM process finished."
echo "--------------------------------------------------"
echo "Pipeline completed successfully."