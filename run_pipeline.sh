#!/bin/bash

# This script automates the entire pipeline:
# 1. Checks for conda environment and required dependencies
# 2. Checks for and downloads the required LLM model if it's missing
# 3. Runs the transcription script (ref_x_max.py)
# 4. Runs the vLLM analysis script (ref_x_vllm.py)

# --- Configuration ---
MODEL_DIR="model"
MODEL_PATH="$MODEL_DIR/LLM.xyz"
DOWNLOAD_URL="https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf"
TEMP_MODEL_NAME="$MODEL_DIR/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf"
REQUIRED_ENV_NAME="audio_pipeline"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# --- 1. Environment and Dependency Checks ---
print_status "Checking environment and dependencies..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    print_error "Conda is not installed or not in PATH."
    print_error "Please install Conda (Miniconda or Anaconda) first."
    print_error "See README.md for installation instructions."
    exit 1
fi

# Check if we're in the correct conda environment
if [[ "$CONDA_DEFAULT_ENV" != "$REQUIRED_ENV_NAME" ]]; then
    print_error "You are not in the required conda environment: $REQUIRED_ENV_NAME"
    print_error "Please activate the environment first:"
    print_error "  conda activate $REQUIRED_ENV_NAME"
    print_error ""
    print_error "If the environment doesn't exist, create it first:"
    print_error "  conda create -n $REQUIRED_ENV_NAME python=3.9"
    print_error "  conda activate $REQUIRED_ENV_NAME"
    print_error "  conda install -c conda-forge ffmpeg=6.1.1"
    print_error "  pip install -r requirements.txt"
    exit 1
fi

print_status "Conda environment '$REQUIRED_ENV_NAME' is active."

# Check if FFmpeg is installed and accessible
if ! command -v ffmpeg &> /dev/null; then
    print_error "FFmpeg is not installed or not accessible."
    print_error "Please install FFmpeg 6.1.1 using:"
    print_error "  conda install -c conda-forge ffmpeg=6.1.1"
    print_error "Or see README.md for alternative installation methods."
    exit 1
fi

# Check FFmpeg version
FFMPEG_VERSION=$(ffmpeg -version 2>&1 | head -n1 | grep -oP 'ffmpeg version \K[0-9]+\.[0-9]+\.[0-9]+')
print_status "FFmpeg version: $FFMPEG_VERSION"

# Warn if not exactly version 6.1.1 but continue
if [[ "$FFMPEG_VERSION" != "6.1.1" ]]; then
    print_warning "FFmpeg version is $FFMPEG_VERSION, but 6.1.1 is recommended."
    print_warning "Continuing anyway... (compatibility may vary)"
fi

# Check if Python dependencies are installed
print_status "Checking Python dependencies..."
if ! python -c "import whisper_timestamped, vllm" &> /dev/null; then
    print_error "Some required Python packages are missing."
    print_error "Please install dependencies using:"
    print_error "  pip install -r requirements.txt"
    exit 1
fi

# Check if required directories exist
REQUIRED_DIRS=("data/80audio/lasthour_transcribe file" "question" "output_multi8" "output_vllm" "logs")
for dir in "${REQUIRED_DIRS[@]}"; do
    if [[ ! -d "$dir" ]]; then
        print_warning "Creating missing directory: $dir"
        mkdir -p "$dir"
    fi
done

# Check if question files exist
QUESTION_FILES=("question/questions_CRS.txt" "question/questions_CRS_2.txt")
for file in "${QUESTION_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        print_warning "Question file not found: $file"
        print_warning "Creating empty placeholder file..."
        touch "$file"
    fi
done

print_status "Environment checks completed successfully."
echo "--------------------------------------------------"

# --- 2. Model Check and Download ---
print_status "Checking for LLM model..."

# Check if the model file already exists
if [[ ! -f "$MODEL_PATH" ]]; then
    print_warning "Model file not found at $MODEL_PATH."
    print_status "Starting download... (This may take a while depending on your network speed)"
    
    # Create the model directory if it doesn't exist
    mkdir -p "$MODEL_DIR"
    
    # Check if wget is available
    if ! command -v wget &> /dev/null; then
        print_error "wget is not installed. Please install wget or download the model manually."
        print_error "Download URL: $DOWNLOAD_URL"
        print_error "Save as: $MODEL_PATH"
        exit 1
    fi
    
    # Download the model using wget with progress bar
    wget --progress=bar:force:noscroll -O "$TEMP_MODEL_NAME" "$DOWNLOAD_URL"
    
    # Check if the download was successful
    if [[ $? -eq 0 ]]; then
        print_status "Download complete."
        # Rename the downloaded file to the expected name
        mv "$TEMP_MODEL_NAME" "$MODEL_PATH"
        print_status "Model successfully set up at $MODEL_PATH."
    else
        print_error "Model download failed. Please check the URL or your network connection."
        # Clean up partial download
        [[ -f "$TEMP_MODEL_NAME" ]] && rm "$TEMP_MODEL_NAME"
        exit 1
    fi
else
    print_status "Model file already exists. Skipping download."
fi

echo "--------------------------------------------------"

# --- 3. Run Transcription Script ---
print_status "Starting the transcription process (ref_x_max.py)..."

# Check if the script exists
if [[ ! -f "ref_x_max.py" ]]; then
    print_error "ref_x_max.py script not found in current directory."
    exit 1
fi

python ref_x_max.py

# Check if the transcription script ran successfully
if [[ $? -ne 0 ]]; then
    print_error "The transcription script (ref_x_max.py) failed. Aborting pipeline."
    exit 1
fi

print_status "Transcription process finished successfully."
echo "--------------------------------------------------"

# --- 4. Verify Transcription Output ---
TRANSCRIPTION_OUTPUT="output_multi8/final_results_lasthour.json"
if [[ ! -f "$TRANSCRIPTION_OUTPUT" ]]; then
    print_error "Expected transcription output file not found: $TRANSCRIPTION_OUTPUT"
    print_error "The transcription script may have failed silently."
    exit 1
fi

print_status "Transcription output verified: $TRANSCRIPTION_OUTPUT"

# --- 5. Run vLLM Script ---
print_status "Starting the vLLM process (ref_x_vllm.py)..."

# Check if the script exists
if [[ ! -f "ref_x_vllm.py" ]]; then
    print_error "ref_x_vllm.py script not found in current directory."
    exit 1
fi

python ref_x_vllm.py \
    --input "$TRANSCRIPTION_OUTPUT" \
    --output output_vllm/ \
    --model "$MODEL_PATH"

if [[ $? -ne 0 ]]; then
    print_error "The vLLM script (ref_x_vllm.py) failed."
    exit 1
fi

print_status "vLLM process finished successfully."
echo "--------------------------------------------------"

# --- 6. Final Verification ---
FINAL_OUTPUT="output_vllm/final_llm_results_lasthour.json"
if [[ -f "$FINAL_OUTPUT" ]]; then
    FINAL_SIZE=$(du -h "$FINAL_OUTPUT" | cut -f1)
    print_status "Final output generated: $FINAL_OUTPUT (Size: $FINAL_SIZE)"
else
    print_warning "Expected final output file not found: $FINAL_OUTPUT"
    print_warning "Check the vLLM script output for details."
fi

print_status "Pipeline completed successfully!"
print_status "Check the logs/ directory for detailed execution logs."
echo "=================================================="
