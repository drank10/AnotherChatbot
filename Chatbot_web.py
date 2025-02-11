import time
import torch
import numpy as np
import pygame
import pyaudio
import wave
import whisper
from openai import OpenAI
# from TTS.api import TTS
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
from PIL import Image, ImageTk
import logging
import json
import os
import configparser
from diffusers import StableDiffusionImg2ImgPipeline,DPMSolverMultistepScheduler,StableDiffusionPipeline
from diffusers.utils import load_image
from peft import PeftModel, LoraConfig
import speech_recognition as sr
from safetensors.torch import load_file
import base64
import requests
import subprocess
import re
from flask import Flask, send_file, render_template_string

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check for CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# Initialize OpenAI client
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Initialize Whisper model
whisper_model = whisper.load_model("large-v3-turbo").to(device)

# Initialize PyAudio
p = pyaudio.PyAudio()

# Global variables
running = True
paused = False
stop_ai_response = False
default_system_prompt = "You are a helpful AI chatbot. Keep responses short and conversational."
audio_samples = {
    "How do you feel?": "great.wav",
    "What's your favorite color?": "blue.wav",
    "Do you like music?": "music.wav"
}

current_image_path = None

# Available TTS models
tts_models = {
    "Default": {
        "ref_audio": "test1.wav",
        "ref_text": "This is the text in the test1.wav file"
    },
    "Julia": {
        "ref_audio": "Julia.wav",
        "ref_text": "This is the text in the Julia.wav file"
    },
    "Kim": {
        "ref_audio": "Kim.wav",
        "ref_text": "This is the text in the Kim.wav file"
    }
}

# Audio recording parameters
silence_threshold = 500
silence_duration = 1.0
max_duration = 10

# New global variables for multi-agent support
agents = {}
current_agent = None

# Initialize Flask app
app = Flask(__name__)
last_image_path = None  # This will store the path to the last generated image

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>AI Chatbot Image Display</title>
    <meta http-equiv="refresh" content="10"> <!-- Refresh every 10 seconds -->
</head>
<body>
    <h1>Last Generated Image:</h1>
    {% if image_path %}
        <img src="{{ url_for('get_image') }}" alt="Generated Image">
    {% else %}
        <p>No image generated yet.</p>
    {% endif %}
</body>
</html>
''', image_path=last_image_path)

@app.route('/image')
def get_image():
    global last_image_path
    if last_image_path is None or not os.path.exists(last_image_path):
        return "No image available", 404
    return send_file(last_image_path, mimetype='image/jpeg')

# Function to run the Flask app in a separate thread
def run_flask_app():
    app.run(port=5000)

# Start the Flask app in a new thread so it doesn't block the main program
flask_thread = threading.Thread(target=run_flask_app)
flask_thread.daemon = True
flask_thread.start()


class F5TTS:
    def __init__(self, model_path="F5-TTS", ref_audio_path=None, ref_text=None):
        self.model_path = model_path
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text

    def tts_to_file(self, text, file_path='output.wav'):
        try:
            # Ensure we have reference audio and text
            if not self.ref_audio_path or not self.ref_text:
                raise ValueError("Reference audio and text must be set for F5 TTS")

            # Construct the F5 TTS command
            command = [
                'f5-tts_infer-cli',
                '--model', self.model_path,
                '--ref_audio', self.ref_audio_path,
                '--ref_text', self.ref_text,
                '--gen_text', text,
                '--output_file', "output.wav",
                '--output_dir', ".",
                '--nfe_step', "12",
                '--load_vocoder_from_local'
            ]

            # Run the F5 TTS command
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"F5 TTS failed: {result.stderr}")

            return True

        except Exception as e:
            logging.error(f"Error in F5 TTS generation: {e}")
            return False

# Initialize F5 TTS with default voice
current_tts_model = "F5-TTS"
tts = F5TTS(
    model_path="F5-TTS",
    ref_audio_path="Default.wav",  # Default reference audio
    ref_text="This is the text said in Default.wav."  # Default reference text
)

class Agent:
    def __init__(self, name, system_prompt, tts_model="Default", template_image=None):
        self.name = name
        self.system_prompt = system_prompt
        self.conversation = []
        self.tts_model = tts_model
        self.template_image = template_image

    def add_message(self, role, content):
        self.conversation.append({"role": role, "content": content})

    def get_conversation_history(self):
        return self.conversation

    def clear_conversation(self):
        self.conversation = []

def clear_conversation():
    global current_agent
    if current_agent:
        current_agent.clear_conversation()
        display_conversation()
        update_status(f"Conversation cleared for agent: {current_agent.name}")
    else:
        update_status("No agent selected. Please select or create an agent first.")

def play_audio(file_path):
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        logging.error(f"Error playing audio: {e}")
    finally:
        pygame.mixer.music.unload()

def play_sample(question):
    if question in audio_samples:
        play_audio(audio_samples[question])
    else:
        update_status("No audio sample available for this question.")

def record_audio():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    update_status("Listening...")
    frames = []
    silent_chunks = 0
    voice_detected = False
    start_time = time.time()

    try:
        while running and not paused:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            
            # Check for voice activity
            audio_data = np.frombuffer(data, dtype=np.int16)
            volume = np.abs(audio_data).mean()
            update_volume_meter(volume)
            
            if volume > silence_threshold.get():
                voice_detected = True
                silent_chunks = 0
            else:
                silent_chunks += 1

            # Stop conditions
            if voice_detected and silent_chunks > int(silence_duration.get() * RATE / CHUNK):
                break
            if time.time() - start_time > max_duration.get():
                break

    except Exception as e:
        logging.error(f"Error during recording: {e}")
        return None
    finally:
        stream.stop_stream()
        stream.close()

    if not voice_detected:
        update_status("No speech detected.")
        return None

    update_status("Done recording.")
    audio = np.frombuffer(b''.join(frames), dtype=np.int16)
    return audio.astype(np.float32) / 32768.0

def remove_emojis(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def get_ai_response(prompt):
    global current_agent, stop_ai_response
    try:
        progress_window = create_progress_window("Getting AI Response")
        messages = [
            {"role": "system", "content": current_agent.system_prompt},
            *current_agent.get_conversation_history(),
            {"role": "user", "content": prompt}
        ]
        
        # Use streaming for the API call
        stream = client.chat.completions.create(
            model="chatwaifu_magnum_v0.2",
            messages=messages,
            temperature=0.7,
            stream=True,
        )
        
        response = ""
        for chunk in stream:
            if stop_ai_response:
                break
            if chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content
        
        progress_window.destroy()
        
        if not stop_ai_response:
            # Remove emojis from the response
            response = remove_emojis(response)
            current_agent.add_message("user", prompt)
            current_agent.add_message("assistant", response)
        
        stop_ai_response = False  # Reset the flag
        return response
    except Exception as e:
        logging.error(f"Error getting AI response: {e}")
        return "I'm sorry, I couldn't process that request. Could you please try again?"



def stop_ai_generation():
    global stop_ai_response
    stop_ai_response = True
    update_status("AI response generation stopped. Say 'stop AI' again to allow responses.")

def update_status(message):
    status_var.set(message)
    root.update_idletasks()

def update_conversation(message, speaker="You"):
    conversation_text.config(state=tk.NORMAL)
    conversation_text.insert(tk.END, f"{speaker}: {message}\n\n")
    conversation_text.config(state=tk.DISABLED)
    conversation_text.see(tk.END)

def set_system_prompt():
    new_prompt = system_prompt_entry.get()
    if current_agent:
        current_agent.system_prompt = new_prompt
        update_status(f"System prompt updated for agent: {current_agent.name}")
    else:
        update_status("No agent selected. Please select or create an agent first.")

def exit_program():
    global running
    running = False
    root.quit()

def add_audio_sample():
    question = audio_question_entry.get()
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if file_path:
        audio_samples[question] = file_path
        update_audio_samples_list()
        update_status(f"Audio sample added for: {question}")

def update_audio_samples_list():
    audio_samples_list.delete(0, tk.END)
    for question, file_path in audio_samples.items():
        audio_samples_list.insert(tk.END, f"{question}: {file_path}")

def change_tts_voice(event):
    global current_agent
    if current_agent:
        selected_voice = voice_var.get()
        if selected_voice != current_agent.tts_model:
            progress_window = create_progress_window("Changing TTS Voice")
            current_agent.tts_model = selected_voice
            update_tts_model()
            progress_window.destroy()
            update_status(f"TTS voice changed to: {current_agent.tts_model} for agent: {current_agent.name}")
    else:
        update_status("No agent selected. Please select or create an agent first.")

def update_tts_model():
    global tts, current_agent
    if current_agent and current_agent.tts_model in tts_models:
        voice_config = tts_models[current_agent.tts_model]
        tts = F5TTS(
            model_path="F5-TTS",
            ref_audio_path=voice_config["ref_audio"],
            ref_text=voice_config["ref_text"]
        )

def create_progress_window(title):
    progress_window = tk.Toplevel(root)
    progress_window.title(title)
    progress_window.geometry("300x100")
    progress_label = ttk.Label(progress_window, text="Processing...")
    progress_label.pack(pady=20)
    progress_bar = ttk.Progressbar(progress_window, mode="indeterminate")
    progress_bar.pack(pady=10)
    progress_bar.start()
    return progress_window

def update_volume_meter(volume):
    normalized_volume = min(volume / 2000, 1.0)  # Adjust the divisor as needed
    volume_meter['value'] = normalized_volume * 100

def save_agents():
    file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
    if file_path:
        data = {name: {"system_prompt": agent.system_prompt, "conversation": agent.conversation, "tts_model": agent.tts_model, "template_image": agent.template_image} 
                for name, agent in agents.items()}
        with open(file_path, 'w') as f:
            json.dump(data, f)
        update_status("Agents saved successfully.")

def load_agents():
    global agents, current_agent
    file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
    if file_path:
        with open(file_path, 'r') as f:
            data = json.load(f)
        agents = {name: Agent(name, info["system_prompt"], info["tts_model"], info["template_image"]) for name, info in data.items()}
        for name, info in data.items():
            agents[name].conversation = info["conversation"]
        update_agent_list()
        current_agent = None
        update_status("Agents loaded successfully.")
        
        # Select the first agent in the list and display its image
        if agents:
            first_agent_name = next(iter(agents))
            agent_listbox.selection_set(0)
            select_agent(None)  # Simulate selection of the first agent

def save_conversation():
    if not current_agent:
        messagebox.showwarning("No Agent Selected", "Please select an agent before saving the conversation.")
        return
    file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
    if file_path:
        data = {
            "agent_name": current_agent.name,
            "system_prompt": current_agent.system_prompt,
            "conversation": current_agent.conversation,
            "tts_model": current_agent.tts_model,
            "template_image": current_agent.template_image
        }
        with open(file_path, 'w') as f:
            json.dump(data, f)
        update_status("Conversation saved successfully.")

def load_conversation():
    global current_agent
    file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
    if file_path:
        with open(file_path, 'r') as f:
            data = json.load(f)
        agent_name = data["agent_name"]
        if agent_name not in agents:
            agents[agent_name] = Agent(agent_name, data["system_prompt"], data["tts_model"], data["template_image"])
        current_agent = agents[agent_name]
        current_agent.system_prompt = data["system_prompt"]
        current_agent.conversation = data["conversation"]
        current_agent.tts_model = data["tts_model"]
        current_agent.template_image = data["template_image"]
        update_agent_list()
        display_conversation()
        update_tts_model()
        update_status("Conversation loaded successfully.")

def create_new_agent():
    name = agent_name_entry.get()
    if name and name not in agents:
        template_image = filedialog.askopenfilename(title="Select template image (optional)", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        agents[name] = Agent(name, default_system_prompt, current_tts_model, template_image)
        update_agent_list()
        select_agent(name)
        update_status(f"New agent created: {name}")
    else:
        messagebox.showwarning("Invalid Name", "Please enter a unique name for the new agent.")

def update_agent_list():
    agent_listbox.delete(0, tk.END)
    for name in agents.keys():
        agent_listbox.insert(tk.END, name)

def select_agent(event):
    global current_agent
    selection = agent_listbox.curselection()
    if selection:
        agent_name = agent_listbox.get(selection[0])
        current_agent = agents[agent_name]
        system_prompt_entry.delete(0, tk.END)
        system_prompt_entry.insert(0, current_agent.system_prompt)
        voice_var.set(current_agent.tts_model)
        update_tts_model()
        display_conversation()
        update_status(f"Selected agent: {agent_name}")
        
        # Display the agent's template image if it exists
        if current_agent.template_image:
            display_generated_image(current_agent.template_image)
        else:
            display_generated_image(None)  # Clear the image display if no template image

def display_conversation():
    conversation_text.config(state=tk.NORMAL)
    conversation_text.delete("1.0", tk.END)
    if current_agent:
        for message in current_agent.get_conversation_history():
            role = "You" if message["role"] == "user" else "AI"
            conversation_text.insert(tk.END, f"{role}: {message['content']}\n\n")
    conversation_text.config(state=tk.DISABLED)
    conversation_text.see(tk.END)

def toggle_pause():
    global paused
    paused = not paused
    if paused:
        pause_button.config(text="Resume")
        update_status("Conversation paused. Say 'pause' again to resume.")
    else:
        pause_button.config(text="Pause")
        update_status("Conversation resumed.")

def change_agent_by_voice(transcription):
    global current_agent
    if transcription.lower().startswith("Can I speak to"):
        agent_name = transcription[10:].strip().lower()
        for name, agent in agents.items():
            if name.lower() == agent_name:
                current_agent = agent
                update_status(f"Switched to agent: {name}")
                system_prompt_entry.delete(0, tk.END)
                system_prompt_entry.insert(0, current_agent.system_prompt)
                voice_var.set(current_agent.tts_model)
                update_tts_model()
                display_conversation()
                return True
    return False

def toggle_image_generation():
    global image_generation_enabled
    image_generation_enabled.set(not image_generation_enabled.get())
    status = "enabled" if image_generation_enabled.get() else "disabled"
    update_status(f"Image generation {status}. Say 'image generation' to change.")

def load_lora_weights(pipeline, checkpoint_path, multiplier=1.0):
    # Load LoRA weights
    lora_state_dict = load_file(checkpoint_path)
    
    # Merge weights
    for key in lora_state_dict:
        if 'lora_down' in key:
            up_key = key.replace('lora_down', 'lora_up')
            model_key = key.replace('lora_down.', '').replace('lora_up.', '')
            model_key = model_key.replace('_lora', '')

            if 'text' in model_key:
                layer = pipeline.text_encoder
            else:
                layer = pipeline.unet

            for name, param in layer.named_parameters():
                if name == model_key:
                    down_weight = lora_state_dict[key]
                    up_weight = lora_state_dict[up_key]

                    param.data += multiplier * torch.mm(up_weight, down_weight).to(param.device)

    return pipeline

def generate_agent_image(agent, prompt):
    global last_image_path
    if not image_generation_enabled.get():
        return None

    # Load the Stable Diffusion model from a checkpoint file
    model_path = "model.ckpt"  # Replace with your .ckpt file path
    lora_path = "lora.safetensors"  # Your LoRA file path

    # Retrieve dynamic values from the GUI
    strength = strength_var.get()
    guidance = guidance_var.get()
    num_steps = steps_var.get()

    # Get the image prompt from the user entry field
    user_image_prompt = image_prompt_entry.get()
    combined_prompt = user_image_prompt if user_image_prompt else f"{agent.system_prompt} {prompt}"

    try:
        if agent.template_image:
            # Use StableDiffusionImg2ImgPipeline for image-to-image generation
            pipe = StableDiffusionImg2ImgPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                safety_checker=None,
                requires_safety_checker=False
            )
        else:
            # Use StableDiffusionPipeline for text-to-image generation
            pipe = StableDiffusionPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                safety_checker=None,
                requires_safety_checker=False
            )
        
        pipe = pipe.to("cuda")

        # Load and merge LoRA weights
        pipe = load_lora_weights(pipe, lora_path, multiplier=0.7)  # Adjust multiplier as needed

        if agent.template_image:
            init_image = Image.open(agent.template_image).convert("RGB")
            init_image = init_image.resize((512, 512))

            image = pipe(
                prompt=combined_prompt,
                image=init_image,
                strength=strength,
                guidance_scale=guidance,
                num_inference_steps=num_steps
            ).images[0]
        else:
            image = pipe(
                prompt=combined_prompt,
                guidance_scale=guidance,
                num_inference_steps=num_steps
            ).images[0]

        # Save the generated image
        os.makedirs("agent_images", exist_ok=True)
        image_path = f"agent_images/{agent.name}_{len(agent.conversation)}.png"
        image.save(image_path)
        last_image_path = image_path
        return image_path

    except Exception as e:
        logging.error(f"Error generating image: {e}")
        return None


def display_generated_image(image_path):
    if image_path is None:
        # Clear the image display
        if hasattr(root, 'image_label'):
            root.image_label.config(image='')
        return

    image = Image.open(image_path)
    image.thumbnail((512, 512))  # Resize the image to fit in the GUI
    photo = ImageTk.PhotoImage(image)
    
    # Create or update the image label
    if hasattr(root, 'image_label'):
        root.image_label.configure(image=photo)
        root.image_label.image = photo
    else:
        root.image_label = ttk.Label(image_frame, image=photo)
        root.image_label.image = photo
        root.image_label.pack(pady=10)

def load_config():
    config = configparser.ConfigParser()
    if os.path.exists('config.ini'):
        config.read('config.ini')
        return config
    return None

def save_config():
    config = configparser.ConfigParser()
    config['Audio'] = {
        'silence_threshold': str(silence_threshold.get()),
        'silence_duration': str(silence_duration.get()),
        'max_duration': str(max_duration.get())
    }
    config['TTS'] = {
        'default_voice': current_tts_model
    }
    with open('config.ini', 'w') as configfile:
        config.write(configfile)

def apply_config(config):
    global current_tts_model
    if config:
        silence_threshold.set(config.getint('Audio', 'silence_threshold', fallback=500))
        silence_duration.set(config.getfloat('Audio', 'silence_duration', fallback=1.0))
        max_duration.set(config.getint('Audio', 'max_duration', fallback=10))
        current_tts_model = config.get('TTS', 'default_voice', fallback="Default")

def create_gradient_background(width, height, color1, color2):
    base = Image.new('RGB', (width, height), color1)
    top = Image.new('RGB', (width, height), color2)
    mask = Image.new('L', (width, height))
    mask_data = []
    for y in range(height):
        mask_data.extend([int(255 * (y / height))] * width)
    mask.putdata(mask_data)
    base.paste(top, (0, 0), mask)
    return ImageTk.PhotoImage(base)

def recognize_command(audio):
    recognizer = sr.Recognizer()
    try:
        command = recognizer.recognize_google(audio).lower()
        return command
    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        print("Could not request results from the speech recognition service")
        return None

def get_base_64_img(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def upload_image():
    global current_image_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])
    if file_path:
        current_image_path = file_path
        display_generated_image(file_path)
        update_status(f"Image uploaded: {os.path.basename(file_path)}")

def analyze_image():
    global current_agent, current_image_path
    if current_image_path:
        try:
            base64_image = get_base_64_img(current_image_path)
            
            completion = client.chat.completions.create(
                model="local-model",
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1000,
                stream=True
            )

            analysis = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    analysis += chunk.choices[0].delta.content
            
            # Add the image analysis to the conversation history
            if current_agent:
                current_agent.add_message("user", "Analyze this image.")
                current_agent.add_message("assistant", analysis)
                display_conversation()
            
            update_conversation(analysis, "AI")
            update_status("Image analyzed. You can now discuss the image with the AI.")
        except Exception as e:
            logging.error(f"Error analyzing image: {e}")
            update_status("An error occurred while analyzing the image.")
    else:
        update_status("Please upload an image first.")

def main_loop():
    global running, paused, stop_ai_response
    update_status("Voice Assistant is ready. Start speaking or say 'stop recording' to quit.")
    
    while running:
        if paused:
            time.sleep(0.1)
            continue
        
        try:
            audio = record_audio()
            if audio is None:
                update_status("No speech detected. Please try again or say 'exit' to quit.")
                continue

            progress_window = create_progress_window("Transcribing Audio")
            result = whisper_model.transcribe(audio)
            progress_window.destroy()
            
            user_prompt = result["text"].strip()
            
            if "pause" in user_prompt.lower():
                toggle_pause()
                continue
            elif "stop ai" in user_prompt.lower():
                stop_ai_generation()    
                continue
            elif "image generation" in user_prompt.lower():
                toggle_image_generation()
                continue
            
            if change_agent_by_voice(user_prompt):
                continue
            
            update_conversation(user_prompt, "You")

            if "stop recording" in user_prompt.lower():
                update_status("Goodbye!")
                running = False
                break

            if user_prompt in audio_samples:
                update_status(f"Playing audio sample for: {user_prompt}")
                play_sample(user_prompt)
            else:
                if current_agent:
                    stop_ai_response = False  # Reset the flag before generating response
                    answer = get_ai_response(user_prompt)
                    if not stop_ai_response:  # Only process the answer if it wasn't stopped
                        update_conversation(answer, "AI")
                        
                        # Generate and display the image only if image generation is enabled
                        if image_generation_enabled.get():
                            image_prompt = f"{current_agent.name} in the style of {current_agent.system_prompt}: {answer}"
                            image_path = generate_agent_image(current_agent, image_prompt)
                            display_generated_image(image_path)
                        else:
                            display_generated_image(None)  # Clear the image display
                        
                        progress_window = create_progress_window("Generating Speech")
                        tts.tts_to_file(text=answer, file_path='output.wav')
                        progress_window.destroy()
                        play_audio('output.wav')
                else:
                    update_status("No agent selected. Please select or create an agent first.")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            update_status("An error occurred. Let's try again.")

    p.terminate()
    update_status("Thank you for using the voice assistant!")

# Create the main window
root = tk.Tk()
root.title("AI Chatbot Interface")
root.geometry("1400x900")
root.configure(bg="#f0f0f0")

image_generation_enabled = tk.BooleanVar(value=True)
strength_var = tk.DoubleVar(value=0.2)
guidance_var = tk.DoubleVar(value=7.5)
steps_var = tk.IntVar(value=10)

# Create a style
style = ttk.Style()
style.theme_use("clam")

# Create gradient background
background_image = create_gradient_background(1400, 900, "#2C3E50", "#34495E")
background_label = tk.Label(root, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Update the style
style = ttk.Style()
style.theme_use("clam")
style.configure("TFrame", background="#34495E")
style.configure("TLabelframe", background="#34495E", foreground="#ECF0F1")
style.configure("TLabelframe.Label", background="#34495E", foreground="#ECF0F1")
style.configure("TLabel", background="#34495E", foreground="#ECF0F1")
style.configure("TButton", background="#627f93", foreground="#ECF0F1")
style.map("TButton", background=[("active", "#3498DB")])
style.configure("TEntry", fieldbackground="#ECF0F1", foreground="#2C3E50")
style.configure("TCombobox", fieldbackground="#ECF0F1", foreground="#2C3E50")
style.configure("Vertical.TScrollbar", background="#627f93", troughcolor="#34495E")

# Create a PanedWindow for the split view
main_paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
main_paned.pack(fill=tk.BOTH, expand=True)

# Left Frame (Conversation and Image)
left_frame = ttk.Frame(main_paned, padding="10")
main_paned.add(left_frame, weight=1)

# Right Frame (Settings and Controls)
right_frame = ttk.Frame(main_paned, padding="10")
main_paned.add(right_frame, weight=1)

# Create a vertical PanedWindow for the left frame
left_paned = ttk.PanedWindow(left_frame, orient=tk.VERTICAL)
left_paned.pack(fill=tk.BOTH, expand=True)

# Header
header_frame = ttk.Frame(left_paned, padding="10", style="TFrame")
left_paned.add(header_frame, weight=0)

title_label = ttk.Label(header_frame, text="AI Chatbot", font=("Helvetica", 24, "bold"))
title_label.pack(side=tk.LEFT)

# Conversation
conversation_frame = ttk.Frame(left_paned, padding="10", style="TFrame")
left_paned.add(conversation_frame, weight=1)

conversation_text = scrolledtext.ScrolledText(conversation_frame, wrap=tk.WORD, width=60, height=10, font=("Helvetica", 10), bg="#2C3E50", fg="#ECF0F1")
conversation_text.pack(fill=tk.BOTH, expand=True)
conversation_text.config(state=tk.DISABLED)

# Image display frame
image_frame = ttk.LabelFrame(left_paned, text="Generated Image", padding="10", style="TLabelframe")
left_paned.add(image_frame, weight=1)

# Status
status_var = tk.StringVar()
status_label = ttk.Label(right_frame, textvariable=status_var, font=("Helvetica", 12))
status_label.pack(pady=10)

# Controls
controls_frame = ttk.LabelFrame(right_frame, text="Controls", padding="10", style="TLabelframe")
controls_frame.pack(fill=tk.X, padx=10, pady=10)

system_prompt_label = ttk.Label(controls_frame, text="System Prompt:", font=("Helvetica", 10))
system_prompt_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

system_prompt_entry = ttk.Entry(controls_frame, width=60, font=("Helvetica", 10))
system_prompt_entry.insert(0, default_system_prompt)
system_prompt_entry.grid(row=0, column=1, columnspan=2, padx=5, pady=5, sticky="ew")

set_prompt_button = ttk.Button(controls_frame, text="Set", command=set_system_prompt, width=10)
set_prompt_button.grid(row=0, column=3, padx=5, pady=5)

voice_label = ttk.Label(controls_frame, text="TTS Voice:", font=("Helvetica", 10))
voice_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")

voice_var = tk.StringVar(value="Default")
voice_dropdown = ttk.Combobox(controls_frame, textvariable=voice_var, values=list(tts_models.keys()), state="readonly", width=30)
voice_dropdown.grid(row=3, column=1, columnspan=2, padx=5, pady=5, sticky="ew")
voice_dropdown.bind("<<ComboboxSelected>>", change_tts_voice)

# Button frame
button_frame = ttk.Frame(controls_frame)
button_frame.grid(row=4, column=0, columnspan=4, pady=10)

pause_button = ttk.Button(button_frame, text="Pause", command=toggle_pause, width=10)
pause_button.pack(side=tk.LEFT, padx=2)

stop_button = ttk.Button(button_frame, text="Stop AI", command=stop_ai_generation, width=10)
stop_button.pack(side=tk.LEFT, padx=2)

clear_conversation_button = ttk.Button(button_frame, text="Clear Chat", command=clear_conversation, width=10)
clear_conversation_button.pack(side=tk.LEFT, padx=2)

save_conversation_button = ttk.Button(button_frame, text="Save Chat", command=save_conversation, width=12)
save_conversation_button.pack(side=tk.LEFT, padx=2)

load_conversation_button = ttk.Button(button_frame, text="Load Chat", command=load_conversation, width=12)
load_conversation_button.pack(side=tk.LEFT, padx=2)

exit_button = ttk.Button(button_frame, text="Exit", command=exit_program, width=12)
exit_button.pack(side=tk.LEFT, padx=2)

# Agent and Audio Sample Management
management_frame = ttk.Frame(right_frame, style="TFrame")
management_frame.pack(fill=tk.X, padx=10, pady=10)

# Agent management
agent_frame = ttk.LabelFrame(management_frame, text="Agents", padding="10", style="TLabelframe")
agent_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

agent_listbox = tk.Listbox(agent_frame, width=20, height=4, font=("Helvetica", 10), bg="#2C3E50", fg="#ECF0F1")
agent_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
agent_listbox.bind('<<ListboxSelect>>', select_agent)

agent_scrollbar = ttk.Scrollbar(agent_frame, orient=tk.VERTICAL, command=agent_listbox.yview)
agent_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
agent_listbox.config(yscrollcommand=agent_scrollbar.set)

agent_control_frame = ttk.Frame(agent_frame, padding="5")
agent_control_frame.pack(fill=tk.X)

agent_name_entry = ttk.Entry(agent_control_frame, width=15, font=("Helvetica", 10))
agent_name_entry.pack(side=tk.LEFT, padx=2)

create_agent_button = ttk.Button(agent_control_frame, text="Create", command=create_new_agent, width=8)
create_agent_button.pack(side=tk.LEFT, padx=2)

save_load_frame = ttk.Frame(agent_frame, padding="5")
save_load_frame.pack(fill=tk.X)

save_agents_button = ttk.Button(save_load_frame, text="Save Agents", command=save_agents, width=12)
save_agents_button.pack(side=tk.LEFT, padx=2, pady=5)

load_agents_button = ttk.Button(save_load_frame, text="Load Agents", command=load_agents, width=12)
load_agents_button.pack(side=tk.LEFT, padx=2, pady=5)

# Audio samples
audio_samples_frame = ttk.LabelFrame(management_frame, text="Audio Samples", padding="10", style="TLabelframe")
audio_samples_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

audio_samples_list = tk.Listbox(audio_samples_frame, width=20, height=4, font=("Helvetica", 10), bg="#2C3E50", fg="#ECF0F1")
audio_samples_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

audio_samples_scrollbar = ttk.Scrollbar(audio_samples_frame, orient=tk.VERTICAL, command=audio_samples_list.yview)
audio_samples_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
audio_samples_list.config(yscrollcommand=audio_samples_scrollbar.set)

add_sample_frame = ttk.Frame(audio_samples_frame, padding="5")
add_sample_frame.pack(fill=tk.X)

audio_question_entry = ttk.Entry(add_sample_frame, width=15, font=("Helvetica", 10))
audio_question_entry.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)

add_sample_button = ttk.Button(add_sample_frame, text="Add", command=add_audio_sample, width=8)
add_sample_button.pack(side=tk.LEFT, padx=2)

update_audio_samples_list()

# Parameters Frame
params_frame = ttk.Frame(right_frame, style="TFrame")
params_frame.pack(fill=tk.X, padx=10, pady=10)

# Audio parameters
audio_params_frame = ttk.LabelFrame(params_frame, text="Audio Parameters", padding="10", style="TLabelframe")
audio_params_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

silence_threshold = tk.IntVar(value=500)
silence_threshold_label = ttk.Label(audio_params_frame, text="Silence Threshold:")
silence_threshold_label.grid(row=0, column=0, padx=5, pady=2, sticky="w")
silence_threshold_entry = ttk.Entry(audio_params_frame, textvariable=silence_threshold, width=8)
silence_threshold_entry.grid(row=0, column=1, padx=5, pady=2)

silence_duration = tk.DoubleVar(value=1.0)
silence_duration_label = ttk.Label(audio_params_frame, text="Silence Duration (s):")
silence_duration_label.grid(row=1, column=0, padx=5, pady=2, sticky="w")
silence_duration_entry = ttk.Entry(audio_params_frame, textvariable=silence_duration, width=8)
silence_duration_entry.grid(row=1, column=1, padx=5, pady=2)

max_duration = tk.IntVar(value=10)
max_duration_label = ttk.Label(audio_params_frame, text="Max Duration (s):")
max_duration_label.grid(row=2, column=0, padx=5, pady=2, sticky="w")
max_duration_entry = ttk.Entry(audio_params_frame, textvariable=max_duration, width=8)
max_duration_entry.grid(row=2, column=1, padx=5, pady=2)

volume_meter_label = ttk.Label(audio_params_frame, text="Mic Volume Meter:")
volume_meter_label.grid(row=3, column=0, padx=5, pady=2, sticky="w")
volume_meter = ttk.Progressbar(audio_params_frame, orient="horizontal", length=100, mode="determinate")
volume_meter.grid(row=3, column=1, padx=5, pady=2, sticky="ew")

# Image parameters
image_params_frame = ttk.LabelFrame(params_frame, text="Image Controls", padding="10", style="TLabelframe")
image_params_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

strength_label = ttk.Label(image_params_frame, text="Strength:")
strength_label.grid(row=0, column=0, padx=5, pady=2, sticky="w")
strength_entry = ttk.Entry(image_params_frame, textvariable=strength_var, width=8)
strength_entry.grid(row=0, column=1, padx=5, pady=2)

guidance_label = ttk.Label(image_params_frame, text="Guidance Scale:")
guidance_label.grid(row=1, column=0, padx=5, pady=2, sticky="w")
guidance_entry = ttk.Entry(image_params_frame, textvariable=guidance_var, width=8)
guidance_entry.grid(row=1, column=1, padx=5, pady=2)

steps_label = ttk.Label(image_params_frame, text="Inference Steps:")
steps_label.grid(row=2, column=0, padx=5, pady=2, sticky="w")
steps_entry = ttk.Entry(image_params_frame, textvariable=steps_var, width=8)
steps_entry.grid(row=2, column=1, padx=5, pady=2)

image_button_frame = ttk.Frame(image_params_frame)
image_button_frame.grid(row=3, column=0, columnspan=2, pady=(10, 0))

toggle_image_gen_button = ttk.Button(image_button_frame, text="ImageGen", command=toggle_image_generation, width=10)
toggle_image_gen_button.pack(side=tk.LEFT, padx=2)

upload_image_button = ttk.Button(image_button_frame, text="Upload Img", command=upload_image, width=12)
upload_image_button.pack(side=tk.LEFT, padx=2)

analyze_image_button = ttk.Button(image_button_frame, text="Analyze Img", command=analyze_image, width=12)
analyze_image_button.pack(side=tk.LEFT, padx=2)

# Image prompt
image_prompt_label = ttk.Label(controls_frame, text="Image Prompt:", font=("Helvetica", 10))
image_prompt_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")

image_prompt_entry = ttk.Entry(controls_frame, width=60, font=("Helvetica", 10))
image_prompt_entry.grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky="ew")

voice_command_frame = ttk.LabelFrame(right_frame, text="Voice Commands", padding="10", style="TLabelframe")
voice_command_frame.pack(fill=tk.X, padx=10, pady=10)

voice_command_label = ttk.Label(voice_command_frame, text="Available voice commands:\n"
                                                          "- 'Pause': Toggle pause/resume\n"
                                                          "- 'Stop AI': Toggle AI response generation\n"
                                                          "- 'Image generation': Enable/disable image generation\n"
                                                          "- 'Audio Sample Name': Plays audio file",
                                font=("Helvetica", 10))
voice_command_label.pack(pady=5)


if __name__ == "__main__":
    # Load configuration
    config = load_config()
    apply_config(config)

    # Start the main loop in a separate thread
    thread = threading.Thread(target=main_loop)
    thread.start()

    # Start the Tkinter event loop
    try:
        root.mainloop()
    finally:
        # Ensure the thread is terminated when the GUI is closed
        running = False
        thread.join()
        
        # Save configuration
        save_config()
        
        logging.info("Application closed.")