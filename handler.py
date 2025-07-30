import os
import torch
import whisper_timestamped as whisper
import runpod
import base64

# --- Model Initialization ---
# This part runs only ONCE when the worker starts.
print("Initializing worker...")

# Check for GPU and set the device
if torch.cuda.is_available():
    DEVICE = "cuda"
    print("GPU found. Using CUDA.")
else:
    DEVICE = "cpu"
    print("Warning: GPU not found. Using CPU.")

# Load the Whisper model into memory.
# This is efficient because it's shared across all requests.
print("Loading Whisper model (turbo)...")
model = whisper.load_model("turbo", device=DEVICE)
print("Model loaded successfully.")

# --- The Handler Function ---
# This function will be called for every API request.
def handler(job):
    """
    Processes a single transcription job.
    """
    job_input = job['input']
    
    # 1. Get audio data from the input
    # We expect the audio file to be sent as a base64 encoded string.
    if 'audio_base64' not in job_input:
        return {"error": "Missing 'audio_base64' in input."}
    
    audio_b64 = job_input['audio_base64']
    
    # Decode the base64 string into bytes and save to a temporary file
    audio_bytes = base64.b64decode(audio_b64)
    temp_audio_path = "temp_audio_file.mp3"
    with open(temp_audio_path, 'wb') as f:
        f.write(audio_bytes)
    
    # 2. Get transcription parameters from the input
    language = job_input.get('language', 'id')
    vad = job_input.get('vad', True)
    detect_disfluencies = job_input.get('detect_disfluencies', True)
    beam_size = job_input.get('beam_size', 5)
    best_of = job_input.get('best_of', 5)

    print(f"Processing job with params: lang={language}, beam_size={beam_size}")

    try:
        # 3. Run the transcription using whisper-timestamped
        result = whisper.transcribe(
            model,
            temp_audio_path,
            vad=vad,
            detect_disfluencies=detect_disfluencies,
            language=language,
            beam_size=beam_size,
            best_of=best_of
        )
        
        print("Transcription successful.")
        return result  # Return the successful result

    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": str(e)}
    finally:
        # 4. Clean up the temporary file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

# --- Start the RunPod server ---
runpod.serverless.start({"handler": handler})
