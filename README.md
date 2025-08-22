**Project Description**
This repository contains a two-stage pipeline for processing audio files. The first script (ref_x_max.py) transcribes audio files in parallel using multiple GPUs. The second script (ref_x_vllm.py) takes the generated transcriptions and performs advanced analysis using a large language model (LLM) with vLLM for high-throughput inference.
The pipeline is designed to be robust, with features like graceful termination (Ctrl+C), detailed logging, and efficient resource management.

**Prerequisites**
Before you begin, ensure the following are installed and configured on the server:
- Python 3.8 or higher
- pip (Python package installer)
- Git
- NVIDIA GPUs with the appropriate CUDA drivers installed. Both scripts are optimized for a multi-GPU environment.

**Setup and Installation**
Follow these steps to set up the project environment and install the necessary dependencies.
1. Clone the Repository
First, clone this repository to your local machine:
**git clone <your-repository-url>**

2. Create a Virtual Environment (Recommended)
It is highly recommended to create a virtual environment to manage the project's dependencies and avoid conflicts with other projects.
**Create the virtual environment
python3 -m venv venv**

Activate the virtual environment
**source venv/bin/activate**

3. Install Required Libraries
Install all the necessary Python libraries using the requirements.txt file. This file contains all dependencies for both scripts.
**pip install -r requirements.txt**


**How to Run the Pipeline**
To execute the entire automated workflow, use the provided bash script **run_pipeline.sh**.
**./run_pipeline.sh**

The script will first run ref_x_max.py to perform audio transcription. Upon its successful completion, it will automatically start ref_x_vllm.py to process the transcriptions. You can monitor the progress in the console and in the logs/ directory.


Scripts Overview
**ref_x_max.py (Transcription Stage)**
- Function: Transcribes stereo MP3 audio files into text.
- Input: Reads .mp3 files from the hardcoded directory: data/80audio/lasthour_transcribe file/.
- Process:
  a. Splits stereo audio into separate channels (Operator and Customer).
  b. Uses the whisper_timestamped model for transcription.
  c. Distributes the workload across multiple GPUs (NUM_GPUS = 4) using multiprocessing (NUM_PROCESSES = 8) for significant speed-up.
  d. Handles graceful shutdown on Ctrl+C, allowing current tasks to finish.
- Output: Creates individual JSON files for each audio in output_multi8/json/ and a final combined file: output_multi8/final_results_lasthour.json.

**ref_x_vllm.py (LLM Analysis Stage)**
- Function: Analyzes the transcribed text to generate summaries, answer specific questions, and determine sentiment.
- Input: Takes the path to the combined JSON file from the first stage (output_multi8/final_results_lasthour.json) as a command-line argument.
- Process:
  a. Uses the vLLM engine for high-performance inference with a large language model.
  b. Utilizes tensor parallelism across multiple GPUs (NUM_GPUS = 4) to run the model.
  c. Processes each transcription with three different prompts (Summary, Q&A, Sentiment).
- Output: Saves a detailed JSON file for each processed transcription in output_vllm/json_llm/ and a final combined results file at output_vllm/final_llm_results_lasthour.json.

**Configuration**
While most settings are within the Python scripts, the following are crucial for deployment:
- Audio Input Path (ref_x_max.py): The script expects audio files to be located in data/80audio/lasthour_transcribe file/. This path is hardcoded.
- Question Files (ref_x_max.py & ref_x_vllm.py): The scripts read questions from question/questions_CRS.txt and question/questions_CRS_2.txt. These files must be present.
- Model Path (run_pipeline.sh): The path to the LLM model (model/LLM.xyz) is specified as an argument in the run_pipeline.sh script. Ensure this path is correct before running.
- GPU Allocation: The number of GPUs and processes are set as constants at the top of each script (NUM_GPUS, NUM_PROCESSES). These can be adjusted based on the server's hardware.

**Logging and Outputs**
- Logs: Both scripts generate detailed logs. They are stored in versioned, monthly directories (e.g., logs/main/202508/ and logs/vllm/202508/). Logs are also printed to the console.
- Intermediate Output: The transcription script's primary output is output_multi8/final_results_lasthour.json.
- Final Output: The vLLM script's final, analyzed output is output_vllm/final_llm_results_lasthour.json.

**Troubleshooting**
If you encounter any issues, please check the following:
  a. Virtual Environment: Ensure you are inside the activated virtual environment (venv). The command prompt should be prefixed with (venv).
  b. Dependencies: Confirm that all libraries in requirements.txt were installed successfully without errors.
  c. Script Permissions: Make sure the run_pipeline.sh script has execute permissions (chmod +x run_pipeline.sh).
  d. File Paths: Double-check that the input/output directories and the model path in run_pipeline.sh are correct and accessible.
  e. CUDA/GPU Issues: Errors like "CUDA out of memory" suggest that the models are too large for the available VRAM. You may need to adjust the gpu_memory_utilization parameter in ref_x_vllm.py or reduce the number of parallel processes in ref_x_max.py. Check that nvidia-smi runs correctly and shows the available GPUs.
