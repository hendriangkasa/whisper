# Project Description
This repository contains a two-stage pipeline for processing audio files. The first script (ref_x_max.py) transcribes audio files in parallel using multiple GPUs. The second script (ref_x_vllm.py) takes the generated transcriptions and performs advanced analysis using a large language model (LLM) with vLLM for high-throughput inference. The pipeline is designed to be robust, with features like graceful termination (Ctrl+C), detailed logging, and efficient resource management.

## Prerequisites
Before you begin, ensure the following are installed and configured on the server:

- Python 3.8 or higher
- pip (Python package installer)
- Git
- **Conda** (Anaconda or Miniconda)
- **FFmpeg 6.1.1** (for audio processing)
- NVIDIA GPUs with the appropriate CUDA drivers installed. Both scripts are optimized for a multi-GPU environment.

## Setup and Installation

### 1. Install Conda (if not already installed)

**For Linux/WSL:**
```bash
# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Make it executable
chmod +x Miniconda3-latest-Linux-x86_64.sh

# Run the installer
./Miniconda3-latest-Linux-x86_64.sh

# Follow the prompts and restart your terminal or run:
source ~/.bashrc
```

**For macOS:**
```bash
# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

# Make it executable
chmod +x Miniconda3-latest-MacOSX-x86_64.sh

# Run the installer
./Miniconda3-latest-MacOSX-x86_64.sh

# Follow the prompts and restart your terminal or run:
source ~/.zshrc  # or ~/.bash_profile depending on your shell
```

**Alternative: Install using package managers**
- **Ubuntu/Debian:** `sudo apt install conda`
- **macOS with Homebrew:** `brew install --cask miniconda`

### 2. Install FFmpeg 6.1.1

**Option A: Using Conda (Recommended)**
```bash
# Create and activate a new conda environment
conda create -n audio_pipeline python=3.9
conda activate audio_pipeline

# Install FFmpeg 6.1.1
conda install -c conda-forge ffmpeg=6.1.1
```

**Option B: System Installation**

*Ubuntu/Debian:*
```bash
# Add FFmpeg PPA for latest versions
sudo add-apt-repository ppa:ubuntuhandbook1/ffmpeg7
sudo apt update
sudo apt install ffmpeg=7:6.1.1*
```

*macOS with Homebrew:*
```bash
# Install specific version (may need to search for available versions)
brew install ffmpeg@6.1
# Link the version
brew link ffmpeg@6.1
```

*CentOS/RHEL/Rocky Linux:*
```bash
# Enable EPEL and RPM Fusion repositories
sudo dnf install epel-release
sudo dnf install --nogpgcheck https://dl.fedoraproject.org/pub/epel/epel-release-latest-$(rpm -E %rhel).noarch.rpm
sudo dnf install --nogpgcheck https://mirrors.rpmfusion.org/free/el/rpmfusion-free-release-$(rpm -E %rhel).noarch.rpm

# Install FFmpeg
sudo dnf install ffmpeg
```

**Verify FFmpeg installation:**
```bash
ffmpeg -version
# Should show version 6.1.1 or compatible
```

### 3. Clone the Repository
First, clone this repository to your local machine:
```bash
git clone <repository-url>
cd <repository-name>
```

### 4. Create and Activate Conda Environment
It is highly recommended to use conda to manage the project's dependencies and avoid conflicts with other projects.

```bash
# Create the conda environment with Python 3.9
conda create -n audio_pipeline python=3.9

# Activate the conda environment
conda activate audio_pipeline

# Install FFmpeg in the environment (if not installed system-wide)
conda install -c conda-forge ffmpeg=6.1.1
```

### 5. Install Required Python Libraries
Install all the necessary Python libraries using the requirements.txt file. This file contains all dependencies for both scripts.
```bash
pip install -r requirements.txt
```

**Note:** Make sure you're in the activated conda environment (`audio_pipeline`) when installing packages. You should see `(audio_pipeline)` at the beginning of your command prompt.

## How to Run the Pipeline
To execute the entire automated workflow, use the provided bash script run_pipeline.sh:
```bash
./run_pipeline.sh
```

The script will:
1. Check for conda environment activation
2. Verify FFmpeg installation
3. Download the LLM model if needed
4. Run ref_x_max.py to perform audio transcription
5. Upon successful completion, automatically start ref_x_vllm.py to process the transcriptions

You can monitor the progress in the console and in the logs/ directory.

## Scripts Overview

### ref_x_max.py (Transcription Stage)
- **Function**: Transcribes stereo MP3 audio files into text.
- **Input**: Reads .mp3 files from the hardcoded directory: `data/80audio/lasthour_transcribe file/`.
- **Process**: 
  - a. Splits stereo audio into separate channels (Operator and Customer).
  - b. Uses the whisper_timestamped model for transcription.
  - c. Distributes the workload across multiple GPUs (NUM_GPUS = 4) using multiprocessing (NUM_PROCESSES = 8) for significant speed-up.
  - d. Handles graceful shutdown on Ctrl+C, allowing current tasks to finish.
- **Output**: Creates individual JSON files for each audio in `output_multi8/json/` and a final combined file: `output_multi8/final_results_lasthour.json`.

### ref_x_vllm.py (LLM Analysis Stage)
- **Function**: Analyzes the transcribed text to generate summaries, answer specific questions, and determine sentiment.
- **Input**: Takes the path to the combined JSON file from the first stage (`output_multi8/final_results_lasthour.json`) as a command-line argument.
- **Process**: 
  - a. Uses the vLLM engine for high-performance inference with a large language model.
  - b. Utilizes tensor parallelism across multiple GPUs (NUM_GPUS = 4) to run the model.
  - c. Processes each transcription with three different prompts (Summary, Q&A, Sentiment).
- **Output**: Saves a detailed JSON file for each processed transcription in `output_vllm/json_llm/` and a final combined results file at `output_vllm/final_llm_results_lasthour.json`.

## Configuration
While most settings are within the Python scripts, the following are crucial for deployment:

- **Audio Input Path** (ref_x_max.py): The script expects audio files to be located in `data/80audio/lasthour_transcribe file/`. This path is hardcoded.
- **Question Files** (ref_x_max.py & ref_x_vllm.py): The scripts read questions from `question/questions_CRS.txt` and `question/questions_CRS_2.txt`. These files must be present.
- **Model Path** (run_pipeline.sh): The path to the LLM model (`model/LLM.xyz`) is specified as an argument in the run_pipeline.sh script. Ensure this path is correct before running.
- **GPU Allocation**: The number of GPUs and processes are set as constants at the top of each script (NUM_GPUS, NUM_PROCESSES). These can be adjusted based on the server's hardware.

## Logging and Outputs

- **Logs**: Both scripts generate detailed logs. They are stored in versioned, monthly directories (e.g., `logs/main/202508/` and `logs/vllm/202508/`). Logs are also printed to the console.
- **Intermediate Output**: The transcription script's primary output is `output_multi8/final_results_lasthour.json`.
- **Final Output**: The vLLM script's final, analyzed output is `output_vllm/final_llm_results_lasthour.json`.

## Troubleshooting
If you encounter any issues, please check the following:

- **Conda Environment**: Ensure you are inside the activated conda environment (`audio_pipeline`). The command prompt should be prefixed with `(audio_pipeline)`.
- **FFmpeg Installation**: Verify FFmpeg is properly installed and accessible by running `ffmpeg -version`.
- **Dependencies**: Confirm that all libraries in requirements.txt were installed successfully without errors.
- **Script Permissions**: Make sure the run_pipeline.sh script has execute permissions (`chmod +x run_pipeline.sh`).
- **File Paths**: Double-check that the input/output directories and the model path in run_pipeline.sh are correct and accessible.
- **CUDA/GPU Issues**: Errors like "CUDA out of memory" suggest that the models are too large for the available VRAM. You may need to adjust the `gpu_memory_utilization` parameter in ref_x_vllm.py or reduce the number of parallel processes in ref_x_max.py. Check that `nvidia-smi` runs correctly and shows the available GPUs.

## Environment Management

To deactivate the conda environment when you're done:
```bash
conda deactivate
```

To reactivate it later:
```bash
conda activate audio_pipeline
```

To remove the environment (if needed):
```bash
conda env remove -n audio_pipeline
```
