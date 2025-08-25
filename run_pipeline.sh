#!/bin/bash

# This script automates the entire pipeline:
# 1. Checks for conda environment and required dependencies
# 2. Checks for and downloads the required LLM model if it's missing
# 3. Runs the transcription script (ref_x_max.py) with GPU monitoring
# 4. Runs the vLLM analysis script (ref_x_vllm.py) with GPU monitoring

# --- Configuration ---
MODEL_DIR="model"
MODEL_PATH="$MODEL_DIR/LLM.xyz"
DOWNLOAD_URL="https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf"
TEMP_MODEL_NAME="$MODEL_DIR/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf"
REQUIRED_ENV_NAME="audio_pipeline"

# GPU Monitoring Configuration
GPU_LOG_INTERVAL=60  # Log GPU stats every 60 seconds (1 minute)
GPU_MONITOR_PID=""   # Will store the background process ID

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_gpu() {
    echo -e "${BLUE}[GPU]${NC} $1"
}

# Function to create timestamped log directory
# create_log_dir() {
#     local log_base_dir="logs/gpu_monitoring"
#     local current_month=$(date +"%Y%m")
#     local log_dir="$log_base_dir/$current_month"
    
#     mkdir -p "$log_dir"
#     echo "$log_dir"
# }

# Function to start GPU monitoring
# start_gpu_monitoring() {
#     local log_dir=$(create_log_dir)
#     local timestamp=$(date +"%Y%m%d_%H%M%S")
#     local gpu_log_file="$log_dir/gpu_stats_${timestamp}.log"
    
#     print_gpu "Starting GPU monitoring (logging every ${GPU_LOG_INTERVAL}s to: $gpu_log_file)"
    
#     (
#         while true; do
#             {
#                 echo "=== GPU Status at $(date) ==="
#                 if command -v nvidia-smi &> /dev/null; then
#                     nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,\
# pcie.link.gen.current,pcie.link.width.current,temperature.gpu,utilization.gpu,\
# utilization.memory,memory.total,memory.free,memory.used,power.draw,power.limit \
# --format=csv,noheader,nounits |
#                     while IFS=',' read -r ts name pci driver pstate pgen pwidth temp gpuutil memutil memtot memfree memused pdraw plimit; do
#                         echo "  Timestamp        : $ts"
#                         echo "  GPU Name         : $name"
#                         echo "  PCI Bus ID       : $pci"
#                         echo "  Driver Version   : $driver"
#                         echo "  Perf State       : $pstate"
#                         echo "  PCIe Gen/Width   : Gen $pgen x$pwidth"
#                         echo "  Temp (°C)        : $temp"
#                         echo "  GPU Util (%)     : $gpuutil"
#                         echo "  Mem Util (%)     : $memutil"
#                         echo "  Mem Total (MiB)  : $memtot"
#                         echo "  Mem Free (MiB)   : $memfree"
#                         echo "  Mem Used (MiB)   : $memused"
#                         echo "  Power Draw (W)   : $pdraw"
#                         echo "  Power Limit (W)  : $plimit"
#                         echo "----------------------------------------------"
#                     done
#                 else
#                     echo "nvidia-smi not available"
#                 fi
#                 echo ""
#             } >> "$gpu_log_file" 2>&1
#             sleep "$GPU_LOG_INTERVAL"
#         done
#     ) &
    
#     GPU_MONITOR_PID=$!
#     echo "$GPU_MONITOR_PID" > /tmp/gpu_monitor_pid.tmp
#     print_gpu "GPU monitoring started (PID: $GPU_MONITOR_PID)"
# }


# # Function to stop GPU monitoring
# stop_gpu_monitoring() {
#     if [[ -n "$GPU_MONITOR_PID" ]] && kill -0 "$GPU_MONITOR_PID" 2>/dev/null; then
#         print_gpu "Stopping GPU monitoring (PID: $GPU_MONITOR_PID)"
#         kill "$GPU_MONITOR_PID" 2>/dev/null
#         wait "$GPU_MONITOR_PID" 2>/dev/null
#     elif [[ -f "/tmp/gpu_monitor_pid.tmp" ]]; then
#         local stored_pid=$(cat /tmp/gpu_monitor_pid.tmp)
#         if [[ -n "$stored_pid" ]] && kill -0 "$stored_pid" 2>/dev/null; then
#             print_gpu "Stopping GPU monitoring (PID: $stored_pid)"
#             kill "$stored_pid" 2>/dev/null
#             wait "$stored_pid" 2>/dev/null
#         fi
#         rm -f /tmp/gpu_monitor_pid.tmp
#     fi
#     print_gpu "GPU monitoring stopped"
# }

# Function to show current GPU status
show_gpu_status() {
    print_gpu "Current GPU Status:"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader | while read line; do
            print_gpu "  $line"
        done
    else
        print_warning "nvidia-smi not available"
    fi
}

# Trap to ensure GPU monitoring is stopped on script exit
cleanup() {
    print_status "Cleaning up..."
    # stop_gpu_monitoring
    exit
}

# Set up signal traps
trap cleanup EXIT INT TERM

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

# Check for NVIDIA GPUs
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    print_status "Found $GPU_COUNT NVIDIA GPU(s)"
    show_gpu_status
else
    print_warning "nvidia-smi not found. GPU monitoring will be limited."
    print_warning "Make sure NVIDIA drivers are installed if you plan to use GPUs."
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
REQUIRED_DIRS=("data/80audio/lasthour_transcribe file" "question" "output_multi8" "output_vllm" "logs" "logs/gpu_monitoring")
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

# # --- 3. Start GPU Monitoring ---
# start_gpu_monitoring

# --- 4. Run Transcription Script ---
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

# --- 5. Verify Transcription Output ---
TRANSCRIPTION_OUTPUT="output_multi8/final_results_lasthour.json"
if [[ ! -f "$TRANSCRIPTION_OUTPUT" ]]; then
    print_error "Expected transcription output file not found: $TRANSCRIPTION_OUTPUT"
    print_error "The transcription script may have failed silently."
    exit 1
fi

print_status "Transcription output verified: $TRANSCRIPTION_OUTPUT"

# --- 6. Run vLLM Script ---
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

# --- 7. Final Verification ---
FINAL_OUTPUT="output_vllm/final_llm_results_lasthour.json"
if [[ -f "$FINAL_OUTPUT" ]]; then
    FINAL_SIZE=$(du -h "$FINAL_OUTPUT" | cut -f1)
    print_status "Final output generated: $FINAL_OUTPUT (Size: $FINAL_SIZE)"
else
    print_warning "Expected final output file not found: $FINAL_OUTPUT"
    print_warning "Check the vLLM script output for details."
fi

# Show final GPU status
print_gpu "Final GPU status after pipeline completion:"
show_gpu_status

print_status "Pipeline completed successfully!"
print_status "Check the logs/ directory for detailed execution logs."
# print_gpu "GPU monitoring logs are saved in logs/gpu_monitoring/$(date +"%Y%m")/"
echo "=================================================="
