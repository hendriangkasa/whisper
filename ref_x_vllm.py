import os
import time
import logging
import json
import re
import signal
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import multiprocessing
import argparse
from vllm import LLM, SamplingParams

# Define constants
LOGGING_DIR = "logs/"  # Base directory to store log files
NUM_GPUS = 4  # Number of GPUs to use for tensor parallelism

# Global flag for termination
terminate_flag = multiprocessing.Value('i', 0)

# Create a logging directory structure based on the current month
month_str = datetime.now().strftime('%Y%m')
log_dir_path = os.path.join(LOGGING_DIR, "vllm", month_str)
os.makedirs(log_dir_path, exist_ok=True)

# Setup logging with current date and time in the filename
current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_filename = f"vllm_log_{current_date}.log"
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

logging.info("VLLM Logging setup complete.")

# Signal handler for graceful termination
def signal_handler(sig, frame):
    logging.info("Received termination signal. Stopping all processes gracefully...")
    # Set the termination flag for all processes
    with terminate_flag.get_lock():
        terminate_flag.value = 1
    logging.info("Waiting for processes to finish current tasks...")

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Kill command

# Load questions
def load_questions():
    try:
        with open('question/questions_CRS.txt', 'r', encoding='utf-8') as file:
            questions = file.read()

        with open('question/questions_CRS_2.txt', 'r', encoding='utf-8') as file:
            questions_2 = file.read()
            
        return questions, questions_2
    except Exception as e:
        logging.error(f"Error loading question files: {e}")
        return "", ""

# Helper function to format questions in English
def format_questions_in_english(questions):
    formatted = ""
    for i, question in enumerate(questions.split('\n'), 1):
        if question.strip():  # Only include non-empty lines
            formatted += f"{question}\n"
    return formatted

# Function to remove timestamps from transcriptions
def remove_timestamps(text):
    # Define regex pattern to match timestamps in square brackets
    pattern = r'\[\d+:\d+ - \d+:\d+\] '
    # Use re.sub to replace matches with an empty string
    return re.sub(pattern, '', text)

# Initialize the vLLM model
def initialize_vllm_model(model_path):
    logging.info(f"Initializing vLLM model from {model_path}")
    try:
        # Set up sampling parameters
        sampling_params = SamplingParams(
            temperature=0,           # Zero temperature for deterministic output
            top_p=0.95,               # Default value
            # top_k=40,
            # presence_penalty=0.1,    
            # frequency_penalty=0.1,  
            max_tokens=1024,         # Maximum tokens to generate
            logprobs=1,              # Include log probabilities
            seed=42                  # Fixed seed for reproducibility
        )
        
        # Initialize vLLM engine
        llm = LLM(
            model='model/LLM.xyz',
            tensor_parallel_size=NUM_GPUS,    # Using specified number of GPUs
            gpu_memory_utilization=0.9,      # Memory control
            max_model_len=8192,               # Context window size
            dtype="auto",                     # Let vLLM choose appropriate dtype
            enforce_eager=True,               # Similar to use_mlock for memory management
            load_format="gguf",               # Specify gguf format
            max_num_seqs=1024                 # Maximum number of sequences
        )
        
        logging.info("vLLM model initialized successfully")
        return llm, sampling_params
    except Exception as e:
        logging.error(f"Failed to initialize vLLM model: {e}")
        return None, None

# Function to process prompts for a single transcription
def process_prompts(text, llm, sampling_params, formatted_questions, formatted_questions_2):
    if terminate_flag.value:
        return None
    
    results = {}
    processed_text = remove_timestamps(text)
    
    # Create the prompts
    prompt_summary = f"""[INST]
            {processed_text}
            
            Based on the conversation between the operator and client provided above, please provide a concise summary in up to 5 sentences (no more than 5). Include details such as who initiated the call, the caller's intent, the client's response to the request, and any notable conclusions from the interaction. Feel free to elaborate on key points for clarity and completeness.
            Let's think step by step.
            
            [/INST]
            """
    
    prompt_qa = f"""[INST]
            
            {processed_text}
            
            Answer the following questions based on the conversation between operator and client above and justify your reasoning:
            
            {formatted_questions}
            
            Do not send the questions and conversations text to me and return only the answer in the format example:
            
            1. Yes. Reason: ..
            2. No. Reason: ..
            n. N/A. Reason ..
            
            [/INST]
            """
    
    prompt_sentiment = f"""[INST]
            
            {processed_text}
            
            Answer the following questions based on the conversation between operator and client above:
            
            {formatted_questions_2}

            Please ensure your answers are clear and supported by specific examples from the conversation.
            Do not send the questions and conversations text to me and return only the answer in the format example:
            
            1. Negative - Inappropriate language (animal words)
            2. Negative - Confused
            
            [/INST]
            """
    
    # Process each prompt and measure individual times
    try:
        # Process summary prompt
        if terminate_flag.value:
            return None
            
        start_time = time.time()
        outputs = llm.generate([prompt_summary], sampling_params)
        result_summary = outputs[0].outputs[0]
        summary_time = round(time.time() - start_time, 2)
        logging.info(f"Summary processing time: {summary_time}s")
        
        # Process QA prompt
        if terminate_flag.value:
            return None
            
        start_time = time.time()
        outputs = llm.generate([prompt_qa], sampling_params)
        result_qa = outputs[0].outputs[0]
        qa_time = round(time.time() - start_time, 2)
        logging.info(f"QA processing time: {qa_time}s")
        
        # Process sentiment prompt
        if terminate_flag.value:
            return None
            
        start_time = time.time()
        outputs = llm.generate([prompt_sentiment], sampling_params)
        result_sentiment = outputs[0].outputs[0]
        sentiment_time = round(time.time() - start_time, 2)
        logging.info(f"Sentiment processing time: {sentiment_time}s")
        
        # Format results similar to the llama_cpp style
        results = {
            'summary': {
                'result': {
                    'choices': [{'text': result_summary.text}],
                    'logprobs': str(result_summary.logprobs) if result_summary.logprobs else None
                },
                'processing_time': summary_time,
                'success': True
            },
            'qa': {
                'result': {
                    'choices': [{'text': result_qa.text}],
                    'logprobs': str(result_qa.logprobs) if result_qa.logprobs else None
                },
                'processing_time': qa_time,
                'success': True
            },
            'sentiment': {
                'result': {
                    'choices': [{'text': result_sentiment.text}],
                    'logprobs': str(result_sentiment.logprobs) if result_sentiment.logprobs else None
                },
                'processing_time': sentiment_time,
                'success': True
            },
            'individual_times': {
                'summary': summary_time,
                'qa': qa_time,
                'sentiment': sentiment_time
            },
            'total_sequential_time': summary_time + qa_time + sentiment_time
        }
        
        # Preview the results
        logging.info(f"Summary: {results['summary']['result']['choices'][0]['text'][:100]}...")
        logging.info(f"QA: {results['qa']['result']['choices'][0]['text'][:100]}...")
        logging.info(f"Sentiment: {results['sentiment']['result']['choices'][0]['text'][:100]}...")
        
    except Exception as e:
        logging.error(f"Error processing prompts: {e}")
        results = {
            'summary': {'result': None, 'processing_time': None, 'success': False},
            'qa': {'result': None, 'processing_time': None, 'success': False},
            'sentiment': {'result': None, 'processing_time': None, 'success': False},
            'error': str(e)
        }
    
    return results

# Function to extract and filter answers from the responses
def extract_and_filter_answers(response, n):
    # Dictionary to store the extracted answers
    answers = {}

    # Iterate over questions 1 to n-1
    for i in range(1, n):
        # Use regex to find the answer for the current question number
        match = re.search(rf'{i}\.\s*(.*?)(?={i+1}\.|$)', response, re.IGNORECASE | re.DOTALL)
        if match:
            # Extract the answer
            answer = match.group(1).strip()
            answers[f"Question {i}"] = answer
        else:
            # If no match is found, set the answer as None
            answers[f"Question {i}"] = None

    return answers

# Function to convert all data structures to serializable format
def convert_serializable(obj):
    if isinstance(obj, pd.Series):
        return obj.tolist() if len(obj) > 1 else obj.iloc[0]
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, '__dict__'):  # For custom classes like Logprob
        try:
            return {k: convert_serializable(v) for k, v in obj.__dict__.items()}
        except:
            return str(obj)  # Fallback to string representation
    elif isinstance(obj, dict):
        return {k: convert_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_serializable(item) for item in obj]
    return obj

# Main processing function
def process_data(input_file, output_dir, model_path):
    logging.info(f"Starting to process data from {input_file}")
    
    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    json_output_dir = os.path.join(output_dir, "json_llm")
    os.makedirs(json_output_dir, exist_ok=True)
    
    # Load questions and format them
    questions, questions_2 = load_questions()
    formatted_questions = format_questions_in_english(questions)
    formatted_questions_2 = format_questions_in_english(questions_2)
    
    # Load input data
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        logging.info(f"Loaded {len(data)} records from {input_file}")
    except Exception as e:
        logging.error(f"Failed to load input file {input_file}: {e}")
        return
    
    # Initialize vLLM model
    llm, sampling_params = initialize_vllm_model(model_path)
    if llm is None or sampling_params is None:
        logging.error("Failed to initialize vLLM model. Exiting.")
        return
    
    # Process each record
    for i, record in enumerate(data):
        if terminate_flag.value:
            logging.info("Termination requested. Stopping processing.")
            break
        
        try:
            audio_name = record['audio_name']
            cleaned_transcription = record['cleaned_transcriptions']
            audio_duration = record['audio_duration']
            
            logging.info(f"Processing record {i+1}/{len(data)}: {audio_name}")
            
            # Process the prompts for this record
            results = process_prompts(
                cleaned_transcription, 
                llm, 
                sampling_params, 
                formatted_questions, 
                formatted_questions_2
            )
            
            if results is None:
                logging.warning(f"No results obtained for {audio_name}. Skipping.")
                continue
            
            # Extract results
            summary_result = results['summary']['result']
            qa_result = results['qa']['result']
            sentiment_result = results['sentiment']['result']
            
            summary_time = results['summary']['processing_time']
            qa_time = results['qa']['processing_time']
            sentiment_time = results['sentiment']['processing_time']
            total_time = summary_time + qa_time + sentiment_time
            
            # Extract texts
            summary_text = summary_result['choices'][0]['text']
            qa_text = qa_result['choices'][0]['text']
            sentiment_text = sentiment_result['choices'][0]['text']
            
            # Create result dictionary
            result_dict = {
                'audio_name': audio_name,
                'audio_duration': audio_duration,
                'cleaned_transcriptions': cleaned_transcription,
                'prompt_id': 1,
                'total_duration': total_time,
                'summary_duration': summary_time,
                'category_duration': qa_time,
                'sentiment_duration': sentiment_time,
                'raw_summary': summary_result,
                'raw_qa': qa_result,
                'raw_sentiment': sentiment_result,
                'sen_answers': extract_and_filter_answers(sentiment_text, 3),
                'qa_answers': extract_and_filter_answers(qa_text, 6)
            }
            
            # Ensure all data is serializable
            result_dict = convert_serializable(result_dict)
            
            # Save this result to its own JSON file
            output_file = os.path.join(json_output_dir, f"{os.path.splitext(audio_name)[0]}.json")
            with open(output_file, 'w') as f:
                json.dump(result_dict, f, indent=2)
            
            logging.info(f"Saved results for {audio_name} to {output_file}")
            
        except Exception as e:
            logging.error(f"Error processing record {i+1}: {e}")
    
    # Collect all processed results
    all_results = []
    for filename in os.listdir(json_output_dir):
        if filename.endswith('.json'):
            try:
                file_path = os.path.join(json_output_dir, filename)
                with open(file_path, 'r') as f:
                    record = json.load(f)
                    
                # Create a simplified version with only the necessary fields
                simplified_record = {
                    'audio_name': record['audio_name'],
                    'audio_duration': record['audio_duration'],
                    'cleaned_transcriptions': record['cleaned_transcriptions'],
                    'prompt_id': record['prompt_id'],
                    'total_duration': record['total_duration'],
                    'summary_duration': record['summary_duration'],
                    'category_duration': record['category_duration'],
                    'sentiment_duration': record['sentiment_duration'],
                    'summary': record['raw_summary']['choices'][0]['text'],
                    'sen_answers': record['sen_answers'],
                    'qa_answers': record['qa_answers']
                }
                
                all_results.append(simplified_record)
                
            except Exception as e:
                logging.error(f"Error reading results from {filename}: {e}")

    # Sort results by audio_name if needed
    all_results.sort(key=lambda x: x['audio_name'])

    # Save the final combined results
    final_output_file = os.path.join(output_dir, "final_llm_results_lasthour.json")
    with open(final_output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    logging.info(f"Successfully saved combined results for {len(all_results)} files to {final_output_file}")
    
    
    # Clean up and release resources
    try:
        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Released model resources")
    except Exception as e:
        logging.error(f"Error releasing resources: {e}")
    
    logging.info("Processing complete")

def main():
    parser = argparse.ArgumentParser(description='Process transcription data with vLLM')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to input JSON file (final_results.json)')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Directory for output files')
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Path to the GGUF model file')
    
    args = parser.parse_args()
    
    # Print usage information
    print("\n======================================================")
    print("vLLM Transcription Processing Tool")
    print("======================================================")
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Model path: {args.model}")
    print(f"Using {NUM_GPUS} GPUs for processing")
    print("Press Ctrl+C to stop processing safely")
    print("======================================================\n")
    
    # Process the data
    process_data(args.input, args.output, args.model)
    
    # Clean up
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt")
        print("Stopping all processing - please wait...")
        with terminate_flag.get_lock():
            terminate_flag.value = 1
        # Give signal handler a chance to run
        time.sleep(1)
        print("Exiting script")
        sys.exit(0)