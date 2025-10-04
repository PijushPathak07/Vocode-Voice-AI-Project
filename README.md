# Vocode Voice AI Demo Project Documentation

## Project Overview

This project implements a real-time voice-based conversational AI using the **Vocode** framework, designed to run entirely on free resources. The system enables users to speak into a microphone, have their speech transcribed, processed by a local language model, and responded to with synthesized speech. Key components include:

- **Transcription:** Deepgram’s free-tier API converts spoken input to text.
- **Language Model:** Hugging Face’s DistilGPT2, a lightweight, open-source model, generates responses locally, ensuring no API costs.
- **Speech Synthesis:** Azure Cognitive Services’ free-tier Text-to-Speech generates spoken responses.
- **Audio Handling:** Local microphone and speaker devices manage input/output, with `ffmpeg` for audio processing.

The project is designed for demonstration purposes, running on a Windows system (tested at `E:\Projects\Blackcoffer Pratice\vocode_demo`) with Python 3.10. It supports casual conversations and can be customized for specific use cases.

## Objectives
- Create a fully functional voice AI demo without incurring costs.
- Demonstrate real-time speech-to-text, text generation, and text-to-speech capabilities.
- Provide a modular, extensible codebase for further experimentation.

## Prerequisites

### Software Requirements
- **Python 3.9 or higher**: Download from [python.org](https://python.org) (free).
- **ffmpeg**: Required for audio processing by `pydub`. Download from [ffmpeg.org](https://ffmpeg.org) or [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) and add to system PATH (e.g., `C:\ffmpeg\bin`).
- **Windows OS**: Tested on Windows, but adaptable to macOS/Linux with minor changes.
- **Microphone and Speakers**: Ensure working audio devices (e.g., `Headset (2- realme Buds Air7)` and `Headphones (2- realme Buds Air7)`).

### API Keys
- **Deepgram API Key**: Sign up at [deepgram.com](https://deepgram.com) for a free-tier key (limited usage, sufficient for demos).
- **Azure Speech Key**: Create a free Azure account at [portal.azure.com](https://portal.azure.com), set up a Speech resource in the "Free F0" tier, and note the key and region (e.g., `eastus`).
- **Security Note**: Regenerate keys after exposure and store them securely in a `.env` file.

### Hardware Recommendations
- **RAM**: Minimum 8GB (16GB recommended for smoother performance with DistilGPT2).
- **CPU**: Standard CPU for DistilGPT2 (GPU optional for faster processing).
- **Audio Devices**: Headset recommended for better audio quality (e.g., `realme Buds Air7`).

## Setup Instructions

1. **Create Project Directory**:
   - Run: `mkdir vocode_demo && cd vocode_demo`.
   - This creates a dedicated folder (e.g., `E:\Projects\Blackcoffer Pratice\vocode_demo`).

2. **Set Up Virtual Environment**:
   - Create: `python -m venv venv`.
   - Activate:
     - Windows: `venv\Scripts\activate`
     - macOS/Linux: `source venv/bin/activate`.

3. **Install Dependencies**:
   - Install required packages:
     ```bash
     pip install vocode transformers torch python-dotenv pydantic-settings sounddevice
     ```
   - These are open-source and free.

4. **Install ffmpeg**:
   - Download the `ffmpeg-release-essentials.zip` from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/).
   - Extract to `C:\ffmpeg`.
   - Add `C:\ffmpeg\bin` to system PATH:
     - Right-click 'This PC' > Properties > Advanced system settings > Environment Variables > Edit `Path` > Add `C:\ffmpeg\bin`.
   - Verify: `ffmpeg -version`.

5. **Configure API Keys**:
   - Create a `.env` file in the project directory:
     ```plaintext
     DEEPGRAM_API_KEY=your_regenerated_deepgram_key
     AZURE_SPEECH_KEY=your_regenerated_azure_key
     AZURE_SPEECH_REGION=eastus
     ```
   - Replace placeholders with regenerated keys (do not use exposed keys).

6. **Save the Code**:
   - Create `vocode_conversation.py` with the code provided earlier (see [Conversation Log](#conversation-log) for context).
   - Ensure the code matches the working version, which uses `BaseAgent` and `LocalLLMAgentConfig`.

7. **Run the Demo**:
   - Activate the virtual environment: `venv\Scripts\activate`.
   - Run: `python vocode_conversation.py`.
   - Select audio devices when prompted (e.g., microphone `0` or `18`, speaker `3` or `18`).
   - Speak to interact; press Ctrl+C to stop.

## Code Explanation

The `vocode_conversation.py` script implements a voice-based conversational AI. Below is a breakdown of its key components:

### Imports and Setup
- **Libraries**:
  - `asyncio`, `signal`: For asynchronous event handling and graceful shutdown.
  - `pydantic_settings`, `dotenv`: For configuration and environment variable management.
  - `vocode.*`: Vocode modules for audio handling, transcription, synthesis, and conversation logic.
  - `transformers`: Hugging Face’s library for the DistilGPT2 model.
  - `sounddevice`: For audio device management.
- **ffmpeg Configuration**:
  ```python
  os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"
  ```
  Ensures `pydub` finds `ffmpeg`.

### Audio Device Configuration
- Lists available devices for debugging:
  ```python
  print("Available audio devices:")
  for i, device in enumerate(sd.query_devices()):
      print(f"{i}: {device['name']} (hostapi: {sd.query_hostapis(device['hostapi'])['name']})")
  ```
- Sets `sounddevice` defaults:
  ```python
  sd.default.device = None
  sd.default.samplerate = 16000
  ```

### Custom Agent Configuration
- **LocalLLMAgentConfig**:
  ```python
  class LocalLLMAgentConfig(AgentConfig, type="agent_local_llm"):
      prompt_preamble: str = "You are a helpful and friendly AI assistant..."
      initial_message_text: str = "Hello! I'm ready to chat..."
  ```
  Defines the prompt and initial message.

### Custom Agent (LocalLLMAgent)
- Subclasses `BaseAgent` for Vocode compatibility:
  ```python
  class LocalLLMAgent(BaseAgent[LocalLLMAgentConfig]):
      def __init__(self, agent_config: LocalLLMAgentConfig):
          super().__init__(agent_config=agent_config)
          self.generator = pipeline("text-generation", model="distilgpt2", device=-1)
  ```
- Methods:
  - `process`: Handles incoming input (transcriptions or strings).
  - `generate_response`: Generates AI responses using DistilGPT2, with conversation history.
  - `clean_response`: Formats responses to avoid fragments and ensure coherence.
  - Maintains a `conversation_history` list to provide context.

### Main Function
- Sets up components:
  - **Microphone/Speaker**: `create_streaming_microphone_input_and_speaker_output(use_default_devices=False)`.
  - **Transcriber**: Deepgram with `PunctuationEndpointingConfig` for responsive speech detection.
  - **Agent**: `LocalLLMAgent` with DistilGPT2.
  - **Synthesizer**: Azure with `en-US-JennyNeural` voice.
- Starts the conversation:
  ```python
  conversation = StreamingConversation(...)
  await conversation.start()
  ```

### Error Handling
- Includes robust error handling in `process` and `generate_response` with fallbacks for failed generations.
- Graceful shutdown on Ctrl+C:
  ```python
  signal.signal(signal.SIGINT, signal_handler)
  ```

## Usage Guide

1. **Run the Script**:
   - Execute `python vocode_conversation.py`.
   - Select microphone and speaker devices from the printed list.

2. **Interact**:
   - Wait for the AI’s greeting: "Hello! I'm ready to chat. What would you like to talk about?"
   - Speak clearly into the microphone.
   - Wait for the AI to respond (transcription takes ~1 second, response generation ~1-2 seconds).
   - Press Ctrl+C to end.

3. **Example Interaction** (from log):
   - User: "I'd like to talk about my life."
   - AI: "This could be something you're familiar with..."
   - User: "What should I do now?"
   - AI: "I know that most people are afraid..."
   - User: "Thanks for I would like to end this conversation."
   - AI: "You don't want me talking right away again anymore anyway..."

4. **Monitor Logs**:
   - Check console for debugging info (e.g., transcription confidence, AI responses).

## Troubleshooting

### Common Issues
- **Audio Device Errors**:
  - If no sound, test devices outside the app (e.g., play music).
  - Try headset devices (e.g., `28: imthethunder`).
  - Run: `python -c "import sounddevice as sd; print(sd.query_devices())"` to verify devices.
- **API Errors**:
  - Verify keys in Deepgram/Azure dashboards.
  - Regenerate keys if exposed and update `.env`.
  - Monitor free-tier limits.
- **Incoherent Responses**:
  - DistilGPT2 may produce off-topic responses. Switch to `gpt2-medium`:
    ```python
    self.generator = pipeline("text-generation", model="gpt2-medium", device=-1)
    ```
  - Refine `prompt_preamble` for better context.
- **Ignored Utterances**:
  - Adjust `time_cutoff_seconds` in `PunctuationEndpointingConfig` to 0.5 for faster response.
  - Enable interruptions:
    ```python
    yield AgentResponseMessage(message=BaseMessage(text=ai_response)), True
    ```
- **Redis Warning**:
  - Ignore for now (caching disabled). Install Redis locally if desired.

### Debugging Tips
- Check console logs for errors (e.g., `[ERROR]` lines).
- Share full error messages and device lists for assistance.
- Test audio with a standalone script:
  ```python
  from pydub import AudioSegment
  print(AudioSegment.ffmpeg)
  ```

## Potential Enhancements

1. **Improve Response Quality**:
   - Use `gpt2-medium` or other Hugging Face models (free, requires more RAM).
   - Enhance prompt engineering:
     ```python
     prompt_preamble="You are a concise, friendly AI. Respond in 1-2 sentences and stay on topic."
     ```

2. **Add Conversation Memory**:
   - Extend `conversation_history` to include more context:
     ```python
     context_items = self.conversation_history[-10:]  # Increase from 8
     ```

3. **Customize Voice**:
   - Change Azure voice:
     ```python
     voice_name="en-US-GuyNeural"
     ```

4. **Reduce Latency**:
   - Lower `time_cutoff_seconds` to 0.5.
   - Optimize DistilGPT2 with `max_new_tokens=30`.

5. **Add Fallback Responses**:
   - Expand `fallback_responses` list for more variety.

## Security Notes
- **API Keys**: Regenerate exposed keys in Deepgram and Azure dashboards. Update `.env` and avoid sharing publicly.
- **Free Tier**: Monitor usage in Deepgram and Azure dashboards to stay within limits.

## Conclusion
This Vocode demo provides a fully functional, cost-free voice AI system using local and free-tier services. It’s extensible for advanced use cases (e.g., larger models, custom prompts) and serves as a solid foundation for exploring voice-based AI applications.