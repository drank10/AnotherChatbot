<img src="AnotherChatbot.jpg">

## Overview
This is an basic AI chatbot interface designed to handle various text-to-speech (TTS) functionalities and image generation based on user interactions.
It uses the F5 TTS engine and can handle responses in multiple emotions 
The LLM can be hosted using LM Studio or Ollama and supports current vision LLM models

## Features
- **Multi-Agent Support:** Switch between different agents with unique personalities and styles.
- **Text-to-Speech Integration:** Generate audio responses from text using various TTS models.
- **Speech-to-Text Integration:** Capture audio input using OpenAI Whisper.
- **Image Generation:** Create visual representations of the chatbot in different styles based on prompt.
- **User-Friendly Interface:** A graphical user interface (GUI) for easy interaction.

## Requirements
- Python 3.x
- Tkinter
- PyTorch
- F5 TTS (https://github.com/SWivid/F5-TTS/tree/main/src/f5_tts)
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/drank10/AIChatbotInterface.git
   cd AIChatbotInterface
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the program:
   ```bash
   python main.py
   ```

## Usage

1. **Select an Agent:** Choose or create a new agent from the interface.
2. **Start Chatting:** Type your message and press enter to get a response.
3. **Pause/Unpause:** Use the 'p' key to pause/unpause the conversation.
4. **Toggle Image Generation:** Use the 'i' key to enable/disable image generation.

## Customization
- You can modify the TTS models, system prompts, and other settings through the interface or by editing the source code.

## Contributing

Feel free to fork this repository and submit pull requests! If you encounter any issues, please create an issue on GitHub.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
```
