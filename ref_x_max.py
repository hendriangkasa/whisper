import os
import time
import logging
import json
import re
import signal
import sys
from configparser import ConfigParser
from pydub import AudioSegment
import whisper_timestamped as whisper
from datetime import datetime
import pandas as pd
import numpy as np
import multiprocessing
import torch
from functools import partial
import concurrent.futures

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Define constants
LOGGING_DIR = "logs/"  # Base directory to store log files
# NUM_PROCESSES = 8  # Increased to 10 processes
# NUM_GPUS = 4  # Define the number of available GPUs

NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
NUM_PROCESSES = NUM_GPUS * 2  # 2 processes per GPU

# Global flag for termination
terminate_flag = multiprocessing.Value('i', 0)

# Create a logging directory structure based on the current month
month_str = datetime.now().strftime('%Y%m')
log_dir_path = os.path.join(LOGGING_DIR, "main", month_str)
os.makedirs(log_dir_path, exist_ok=True)

# Setup logging with current date and time in the filename
current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_filename = f"log_{current_date}.log"  # Example: log_2024-07-03_12-00-00.log
log_path = os.path.join(log_dir_path, log_filename)

# Configure logging
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add console handler for viewing logs in terminal
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logging.info("Logging setup complete.")
logging.info(f"Auto-detected {NUM_GPUS} GPUs, using {NUM_PROCESSES} processes")

# Signal handler for graceful termination
def signal_handler(sig, frame):
    logging.info("Received termination signal. Stopping all processes gracefully...")
    # Set the termination flag for all processes
    with terminate_flag.get_lock():
        terminate_flag.value = 1
    logging.info("Waiting for processes to finish current tasks...")
    # Don't exit immediately, let the main loop handle the termination

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Kill command

# Initialize CUDA devices and set up device assignment
def initialize_gpu_devices():
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        logging.error("No CUDA devices available, falling back to CPU")
        return []
    
    logging.info(f"Found {num_gpus} CUDA devices")
    return list(range(num_gpus))

available_gpus = initialize_gpu_devices()

# Function to load model on specific GPU
def load_model_on_gpu(process_id):
    # Distribute processes evenly across available GPUs
    if available_gpus:
        gpu_id = process_id % len(available_gpus)
        device = f"cuda:{gpu_id}"
    else:
        device = "cpu"
    
    # Clear CUDA cache before loading new model
    if torch.cuda.is_available():
        with torch.cuda.device(int(device.split(':')[1]) if 'cuda' in device else 'cpu'):
            torch.cuda.empty_cache()
    
    logging.info(f"Process {process_id} loading whisper model on device: {device}")
    try:
        with torch.cuda.device(int(device.split(':')[1]) if 'cuda' in device else 'cpu'):
            model = whisper.load_model("turbo", device=device)
        
        # Set lower VRAM usage for models sharing GPUs
        # if hasattr(model, 'set_device_usage'):
        #     model.set_device_usage(0.25)  # Use only 25% of VRAM per model
            
        return model, device
    except Exception as e:
        logging.error(f"Error loading model on {device} for process {process_id}: {e}")
        return None, device

# LLAMA setup
with open('question/questions_CRS.txt', 'r', encoding='utf-8') as file:
    questions = file.read()

with open('question/questions_CRS_2.txt', 'r', encoding='utf-8') as file:
    questions_2 = file.read()

# Helper function to format questions in English
def format_questions_in_english(questions):
    formatted = ""
    for i, question in enumerate(questions.split('\n'), 1):
        formatted += f"{question}\n"
    return formatted

formatted_questions = format_questions_in_english(questions)
formatted_questions_2 = format_questions_in_english(questions_2)

def remove_timestamps(text):
    # Define regex pattern to match timestamps in square brackets
    pattern = r'\[\d+:\d+ - \d+:\d+\] '
    # Use re.sub to replace matches with an empty string
    return re.sub(pattern, '', text)

# Base paths
base_path = "data"
output_path = base_path
error_path = "data/error"

# Ensure the output and error directories exist
os.makedirs(error_path, exist_ok=True)
os.makedirs("output_multi8/json", exist_ok=True)  # Changed directory name to reflect 10 processes

# GPU memory lock to prevent concurrent model loading on the same GPU
# gpu_locks = {i: multiprocessing.Lock() for i in range(NUM_GPUS)}
gpu_locks = {i: multiprocessing.Lock() for i in range(NUM_GPUS)} if NUM_GPUS > 0 else {}

# Function to refine segments based on text
def refine_segments_based_on_text(transcription):
    refined_segments = []

    for segment in transcription['segments']:
        segment_text = segment['text']
        words = segment['words']
        start = segment['start']
        end = segment['end']

        # Split the segment text and words based on [*]
        part_indices = [i for i, word in enumerate(words) if word['text'] == '[*]']
        part_indices = [0] + [i + 1 for i in part_indices] + [len(words)]
        
        for i in range(len(part_indices) - 1):
            part_words = words[part_indices[i]:part_indices[i+1]]
            if part_words:
                part_text = ' '.join(word['text'] for word in part_words if word['text'] != '[*]')
                refined_segments.append({
                    'start': part_words[0]['start'],
                    'end': part_words[-1]['end'],
                    'text': part_text,
                    'words': part_words
                })

    transcription['segments'] = refined_segments
    return transcription

# Function to format transcriptions into a list of segment dictionaries
def format_transcriptions(transcription, speaker):
    formatted_transcriptions = []
    for segment in transcription['segments']:
        formatted_transcriptions.append({
            'start': segment['start'],
            'end': segment['end'],
            'role': speaker,
            'text': segment['text']
        })
    return formatted_transcriptions

# Function to merge operator and customer transcriptions into a single string
def merge_opr_and_rec_transcriptions(opr_transcriptions, rec_transcriptions):
    all_segments = sorted(opr_transcriptions + rec_transcriptions, key=lambda x: x['start'])
    merged_transcriptions = []
    for segment in all_segments:
        start = segment['start']
        end = segment['end']
        role = segment['role']
        text = segment['text']
        merged_transcriptions.append(f"[{start//60:.0f}:{start%60:02.0f} - {end//60:.0f}:{end%60:02.0f}] {role}: {text}")
    return "\n".join(merged_transcriptions)

# Function to merge operator and customer transcriptions into a single string
def merge_all_transcriptions(transcriptions):
    merged_transcriptions = []

    for transcription in transcriptions:
        result_opr = refine_segments_based_on_text(transcription['operator_transcription'])
        result_rec = refine_segments_based_on_text(transcription['customer_transcription'])

        opr_transcriptions = format_transcriptions(result_opr, "Operator")
        rec_transcriptions = format_transcriptions(result_rec, "Client")

        merged_transcription = merge_opr_and_rec_transcriptions(opr_transcriptions, rec_transcriptions)
        merged_transcriptions.append(merged_transcription)

        # Update the transcription dictionary with merged transcription
        transcription['merged_transcription'] = merged_transcription

    return merged_transcriptions

# Function to remove blank conversations from transcriptions
def remove_blank_conversations(transcriptions):
    cleaned_transcriptions = []
    for transcription in transcriptions:
        # Split the transcription into lines
        lines = transcription.split('\n')
        
        # Filter out lines that have blank conversations
        filtered_lines = [line for line in lines if not re.match(r'\[\d{1,2}:\d{2} - \d{1,2}:\d{2}\] (Client|Operator):\s*$', line)]
        
        # Join the lines back into a single string
        cleaned_transcription = '\n'.join(filtered_lines)
        
        # Add the cleaned transcription to the list
        cleaned_transcriptions.append(cleaned_transcription)
    
    return cleaned_transcriptions

# Function to transcribe a single channel
def transcribe_channel(file_path, model, is_operator=True):
    # Check termination flag
    if terminate_flag.value:
        return None, 0
        
    speaker_type = "operator" if is_operator else "customer"
    try:
        start_time = time.time()
        
        # Reduce memory usage by setting a lower beam size and best_of
        result = whisper.transcribe(
            model, 
            file_path, 
            vad="auditok", 
            detect_disfluencies=True,
            language='id', 
            beam_size=5,  # Reduced from 5
            best_of=5,    # Reduced from 5
            temperature=(0.0, 0.2, 0.4, 0.6)  # Reduced temperature range
        )
        transcribe_duration = time.time() - start_time
        return result, transcribe_duration
    except Exception as e:
        logging.error(f"Failed to transcribe {speaker_type} audio: {file_path}. Error: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        return None, 0

# Function to process a single audio file
def process_audio_file(audio_file, process_id):
    # Check termination flag
    if terminate_flag.value:
        logging.info(f"Process {process_id} terminating due to stop signal")
        return None
    
    # Determine which GPU to use
    gpu_id = process_id % NUM_GPUS
    
    # Acquire lock for this GPU
    with gpu_locks[gpu_id]:
        # Load model on the assigned GPU
        model, device = load_model_on_gpu(process_id)
        if model is None:
            logging.error(f"Failed to load model for process {process_id}")
            return None
    
    file_path = os.path.join(base_path, audio_file)
    logging.info(f"Process {process_id} on device {device} processing file: {file_path}")

    # Create unique temporary file names based on process ID and timestamp
    timestamp = int(time.time())
    temp_dir = f"temp_{process_id}_{timestamp}"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Check termination flag again before starting heavy processing
        if terminate_flag.value:
            logging.info(f"Process {process_id} terminating due to stop signal")
            return None
            
        # Load the audio file
        audio = AudioSegment.from_mp3(file_path)
        audio_duration = audio.duration_seconds

        # Split stereo audio into individual channels
        channels = audio.split_to_mono()
        operator_audio = channels[0]
        customer_audio = channels[1]

        operator_file = os.path.join(temp_dir, f"operator_{process_id}_{audio_file}")
        customer_file = os.path.join(temp_dir, f"customer_{process_id}_{audio_file}")

        operator_audio.export(operator_file, format="mp3")
        customer_audio.export(customer_file, format="mp3")

        # Check termination flag before starting transcription
        if terminate_flag.value:
            logging.info(f"Process {process_id} terminating due to stop signal")
            # Clean up temp files
            for temp_file in [operator_file, customer_file]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
            return None

        # Transcribe both channels
        result_opr, transcribe_duration_opr = transcribe_channel(operator_file, model, True)
        
        # Check termination flag between transcriptions
        if terminate_flag.value:
            logging.info(f"Process {process_id} terminating due to stop signal")
            # Clean up temp files
            for temp_file in [operator_file, customer_file]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
            return None
            
        result_rec, transcribe_duration_rec = transcribe_channel(customer_file, model, False)

        # Clean up temp files
        for temp_file in [operator_file, customer_file]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

        # Check if both transcriptions were successful
        if result_opr and result_rec:
            # Calculate transcription durations
            transcribe_duration = transcribe_duration_opr + transcribe_duration_rec

            transcription = {
                'audio_name': audio_file,
                'audio_duration': audio_duration,
                'transcription_duration': transcribe_duration,
                'transcription_duration_opr': transcribe_duration_opr,
                'transcription_duration_rec': transcribe_duration_rec,
                'operator_transcription': result_opr,
                'customer_transcription': result_rec
            }
            
            # Write JSON output
            fout = f"output_multi8/json/{os.path.basename(audio_file).replace('.mp3', '.json')}"
            with open(fout, 'w') as f:
                json.dump(transcription, f)
            
            logging.info(f"Process {process_id} successfully transcribed {audio_file}")
            
            # Explicitly release GPU memory
            with gpu_locks[gpu_id]:
                logging.info(f"Process {process_id} releasing GPU memory")
                del model
                if torch.cuda.is_available():
                    with torch.cuda.device(int(device.split(':')[1]) if 'cuda' in device else 'cpu'):
                        torch.cuda.empty_cache()
            
            return transcription, {
                'audio_name': audio_file,
                'audio_duration': audio_duration
            }
        else:
            logging.error(f"Process {process_id} failed to transcribe {audio_file}")
            
            # Explicitly release GPU memory
            with gpu_locks[gpu_id]:
                logging.info(f"Process {process_id} releasing GPU memory")
                del model
                if torch.cuda.is_available():
                    with torch.cuda.device(int(device.split(':')[1]) if 'cuda' in device else 'cpu'):
                        torch.cuda.empty_cache()
            
            return None
    
    except Exception as e:
        logging.error(f"Process {process_id} encountered error processing {audio_file}: {e}")
        # Clean up temp directory if it exists
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Make sure to release GPU memory even in case of an error
        try:
            with gpu_locks[gpu_id]:
                del model
                if torch.cuda.is_available():
                    with torch.cuda.device(device.split(':')[1] if 'cuda' in device else 'cpu'):
                        torch.cuda.empty_cache()
        except:
            pass
            
        return None

# Function to check if processes are still running
def processes_running(futures):
    return any(not future.done() for future in futures)

# Resource allocation monitoring thread
def monitor_resources():
    while not terminate_flag.value:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
                memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)    # GB
                logging.info(f"GPU {i}: Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
        time.sleep(30)  # Check every 30 seconds

# Main function to process audio files in parallel
def main():
    mp3_files = [f for f in os.listdir(base_path) if f.lower().endswith('.mp3')]
    if not mp3_files:
        logging.info(f"No MP3 files found in {base_path}")
        return
    
    logging.info(f"Found {len(mp3_files)} MP3 files to process")
    
    # Reset termination flag
    with terminate_flag.get_lock():
        terminate_flag.value = 0
    
    # Start resource monitoring in a separate thread
    import threading
    monitor_thread = threading.Thread(target=monitor_resources)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Process files in batches - now we use a larger batch size to utilize all 10 processes
    batch_size = NUM_PROCESSES  # Process NUM_PROCESSES files at a time
    transcriptions = []
    audio_data = []
    
    for batch_start in range(0, len(mp3_files), batch_size):
        batch_files = mp3_files[batch_start:batch_start + batch_size]
        logging.info(f"Processing batch of {len(batch_files)} files ({batch_start+1}-{batch_start+len(batch_files)} of {len(mp3_files)})")
        
        # Create a process pool for this batch
        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
            # Submit jobs with process IDs
            futures = {
                executor.submit(process_audio_file, mp3_file, i % NUM_PROCESSES): mp3_file
                for i, mp3_file in enumerate(batch_files)
            }
            
            try:
                # Collect results for this batch
                for future in concurrent.futures.as_completed(futures):
                    # Check if termination was requested
                    if terminate_flag.value:
                        logging.info("Termination requested, cancelling pending tasks...")
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        break
                    
                    mp3_file = futures[future]
                    try:
                        result = future.result()
                        if result:
                            transcription, audio_datum = result
                            transcriptions.append(transcription)
                            audio_data.append(audio_datum)
                    except Exception as e:
                        logging.error(f"Error processing {mp3_file}: {e}")
            
            except KeyboardInterrupt:
                logging.info("KeyboardInterrupt received in main loop")
                with terminate_flag.get_lock():
                    terminate_flag.value = 1
                
                # Cancel all pending futures
                for f in futures:
                    if not f.done():
                        f.cancel()
                
                break  # Exit the batch loop
        
        # Clear GPU memory after each batch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
        
        # If termination was requested, break out of the batch loop
        if terminate_flag.value:
            break
    
    # Post-process the transcriptions if we have any and termination wasn't requested
    if transcriptions and not terminate_flag.value:
        logging.info("Processing completed transcriptions...")
        merged_transcriptions = merge_all_transcriptions(transcriptions)
        cleaned_transcriptions = remove_blank_conversations(merged_transcriptions)
        
        # Create final data structure
        data = [
            {**audio_data[i], 'cleaned_transcriptions': cleaned_transcriptions[i]}
            for i in range(len(transcriptions))
        ]
        
        # Save the final results
        with open("output_multi8/final_results.json", 'w') as f:
            json.dump(data, f)
        
        logging.info(f"Successfully processed {len(transcriptions)} files")
    else:
        if terminate_flag.value:
            logging.warning("Processing terminated before completion")
        else:
            logging.warning("No transcriptions were successfully created")
    
    # Final status message
    if terminate_flag.value:
        logging.info("Process terminated by user")
    else:
        logging.info("All processing completed")

# Show user how to stop the script
def print_usage():
    print("\n======================================================")
    print("Whisper GPU Multiprocessing Transcription Tool")
    print("======================================================")
    print(f"Running with {NUM_PROCESSES} processes across {NUM_GPUS} GPUs")
    print("Press Ctrl+C to stop all processing safely")
    print("The script will finish current transcriptions and exit")
    print("======================================================\n")

# Entry point for the script
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)  # Important for CUDA compatibility
    print_usage()
    try:
        main()

        import gc
        gc.collect()
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()

        # If using multiprocessing, make sure all processes are terminated
        for p in multiprocessing.active_children():
            p.terminate()
            p.join()

    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt in main process")
        print("Stopping all processing - please wait...")
        with terminate_flag.get_lock():
            terminate_flag.value = 1
        # Give signal handler a chance to run
        time.sleep(1)
        print("Exiting script")
        sys.exit(0)
        
        import gc
        gc.collect()
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()

        # If using multiprocessing, make sure all processes are terminated
        for p in multiprocessing.active_children():
            p.terminate()
            p.join()
