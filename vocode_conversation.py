import asyncio
import signal
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
from vocode.helpers import create_streaming_microphone_input_and_speaker_output
from vocode.logging import configure_pretty_logging
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import AzureSynthesizerConfig
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
)
from vocode.streaming.models.agent import AgentConfig
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.synthesizer.azure_synthesizer import AzureSynthesizer
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber
from vocode.streaming.agent.base_agent import BaseAgent, AgentResponseMessage, AgentInput
from vocode.streaming.models.events import Sender
from transformers import pipeline
from typing import Optional, AsyncGenerator
import sounddevice as sd
import os
import re

# Set ffmpeg path for pydub
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

# Configure sounddevice
sd.default.device = None
sd.default.samplerate = 16000

# Log available audio devices for debugging
print("Available audio devices:")
for i, device in enumerate(sd.query_devices()):
    print(f"{i}: {device['name']} (hostapi: {sd.query_hostapis(device['hostapi'])['name']})")


# Custom AgentConfig for our local LLM
class LocalLLMAgentConfig(AgentConfig, type="agent_local_llm"):
    prompt_preamble: str = "You are a friendly AI assistant having a casual conversation."
    initial_message_text: str = "Hello! I'm ready to chat. What would you like to talk about?"


# Custom agent using Hugging Face's local model
class LocalLLMAgent(BaseAgent[LocalLLMAgentConfig]):
    def __init__(self, agent_config: LocalLLMAgentConfig):
        super().__init__(agent_config=agent_config)
        self.prompt_preamble = agent_config.prompt_preamble
        self.initial_message_text = agent_config.initial_message_text
        self.conversation_history = []
        print("Loading distilgpt2 model...")
        self.generator = pipeline("text-generation", model="distilgpt2", device=-1)
        print("LocalLLMAgent initialized successfully!")
        
    def get_agent_config(self) -> LocalLLMAgentConfig:
        """Return the agent configuration"""
        return self.agent_config
    
    async def process(self, item: AgentInput):
        """Process incoming agent input - required by BaseAgent"""
        try:
            if isinstance(item, str):
                human_input = item
            elif hasattr(item, 'payload') and hasattr(item.payload, 'transcription'):
                human_input = item.payload.transcription.message
            elif hasattr(item, 'transcription'):
                human_input = item.transcription.message
            else:
                print(f"[WARNING] Unknown input type: {type(item)}")
                return
            
            # Skip empty inputs
            if not human_input or not human_input.strip():
                print("[DEBUG] Skipping empty input")
                return
            
            print(f"\n[USER SAID]: '{human_input}'")
            
            # Generate response
            async for response_message in self.generate_response(
                human_input=human_input,
                conversation_id=item.conversation_id if hasattr(item, 'conversation_id') else "default",
                is_interrupt=False
            ):
                self.produce_interruptible_agent_response_event_nonblocking(response_message)
                
        except Exception as e:
            print(f"[ERROR] Error in process: {e}")
            import traceback
            traceback.print_exc()
    
    def clean_response(self, text: str) -> str:
        """Clean and format the generated response"""
        # Remove the prompt and context
        text = text.strip()
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Take first 1-2 complete sentences
        result_sentences = []
        for sent in sentences[:3]:
            sent = sent.strip()
            if sent and len(sent) > 10:  # Avoid very short fragments
                # Ensure sentence ends with punctuation
                if not sent[-1] in '.!?':
                    sent += '.'
                result_sentences.append(sent)
                if len(result_sentences) >= 2:  # Max 2 sentences
                    break
        
        if result_sentences:
            return ' '.join(result_sentences)
        
        # Fallback responses
        return "That's interesting! Could you tell me more?"
    
    async def generate_response(
        self,
        human_input: str,
        conversation_id: str,
        is_interrupt: bool = False,
    ) -> AsyncGenerator[AgentResponseMessage, None]:
        """Generate a response using the local LLM"""
        try:
            print(f"[AI THINKING]...")
            
            # Add to conversation history
            self.conversation_history.append(f"Person: {human_input}")
            
            # Build context from recent history (last 4 exchanges)
            context_items = self.conversation_history[-8:]
            
            # Create a more structured prompt
            prompt = f"""The following is a friendly conversation between a person and an AI assistant.

{chr(10).join(context_items)}
Assistant:"""
            
            # Generate response with better parameters
            generated = self.generator(
                prompt, 
                max_new_tokens=40,  # Limit new tokens generated
                num_return_sequences=1, 
                truncation=True,
                pad_token_id=self.generator.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,  # Lower temperature for more coherent responses
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2  # Discourage repetition
            )[0]["generated_text"]
            
            # Extract just the AI's response (everything after "Assistant:")
            if "Assistant:" in generated:
                ai_response = generated.split("Assistant:")[-1].strip()
            else:
                ai_response = generated[len(prompt):].strip()
            
            # Stop at the next speaker turn if any
            if "Person:" in ai_response:
                ai_response = ai_response.split("Person:")[0].strip()
            
            # Clean the response
            ai_response = self.clean_response(ai_response)
            
            # Fallback if response is too short or repetitive
            if (len(ai_response.strip()) < 10 or 
                human_input.lower() in ai_response.lower()):
                
                fallback_responses = [
                    "That's really interesting! What else can you share about that?",
                    "I'd love to hear more. Can you elaborate?",
                    "Tell me more about that!",
                    "That sounds fascinating. What makes you say that?",
                    "I see. What's your perspective on that?"
                ]
                import random
                ai_response = random.choice(fallback_responses)
            
            print(f"[AI RESPONDS]: '{ai_response}'")
            
            # Add to conversation history
            self.conversation_history.append(f"Assistant: {ai_response}")
            
            # Keep history manageable
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            # Yield the response
            yield AgentResponseMessage(message=BaseMessage(text=ai_response))
            
        except Exception as e:
            print(f"[ERROR] Error generating response: {e}")
            import traceback
            traceback.print_exc()
            yield AgentResponseMessage(
                message=BaseMessage(text="Could you say that again? I didn't quite catch that.")
            )


load_dotenv()
configure_pretty_logging()


class Settings(BaseSettings):
    deepgram_api_key: str
    azure_speech_key: str
    azure_speech_region: str = "eastus"
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"
    )


settings = Settings()


async def main():
    print("\n" + "="*60)
    print("Starting Voice Conversation with Local LLM")
    print("="*60 + "\n")
    
    # Create microphone input and speaker output
    print("Setting up audio devices...")
    microphone_input, speaker_output = create_streaming_microphone_input_and_speaker_output(
        use_default_devices=False
    )
    
    # Create transcriber with adjusted settings
    print("Setting up Deepgram transcriber...")
    transcriber = DeepgramTranscriber(
        DeepgramTranscriberConfig.from_input_device(
            microphone_input,
            endpointing_config=PunctuationEndpointingConfig(
                time_cutoff_seconds=1.0  # Shorter cutoff for more responsive conversation
            ),
            api_key=settings.deepgram_api_key,
            model="nova-2",
            tier=None
        )
    )
    
    # Create agent config
    print("Setting up Local LLM agent...")
    agent_config = LocalLLMAgentConfig(
        prompt_preamble="You are a helpful and friendly AI assistant. Keep your responses brief and conversational.",
        initial_message=BaseMessage(text="Hello! I'm ready to chat. What would you like to talk about?")
    )
    
    # Create agent
    agent = LocalLLMAgent(agent_config=agent_config)
    
    # Create synthesizer
    print("Setting up Azure speech synthesizer...")
    synthesizer = AzureSynthesizer(
        AzureSynthesizerConfig.from_output_device(
            speaker_output,
            voice_name="en-US-JennyNeural"
        ),
        azure_speech_key=settings.azure_speech_key,
        azure_speech_region=settings.azure_speech_region
    )
    
    # Create conversation
    print("Creating conversation...")
    conversation = StreamingConversation(
        output_device=speaker_output,
        transcriber=transcriber,
        agent=agent,
        synthesizer=synthesizer,
        conversation_id="local-llm-test",
        per_chunk_allowance_seconds=0.01,
        speed_coefficient=1.0
    )
    
    # Start conversation
    await conversation.start()
    
    print("\n" + "="*60)
    print("🎤 CONVERSATION STARTED!")
    print("="*60)
    print("\nInstructions:")
    print("1. Wait for the AI to finish speaking")
    print("2. Speak clearly into your microphone")
    print("3. Wait for the AI to respond")
    print("4. Press Ctrl+C to end the conversation")
    print("\nListening for your speech...\n")
    
    # Handle Ctrl+C gracefully
    async def shutdown():
        print("\n\nShutting down conversation...")
        await conversation.terminate()
    
    def signal_handler(sig, frame):
        asyncio.create_task(shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Main loop to receive audio
    try:
        while conversation.is_active():
            chunk = await microphone_input.get_audio()
            conversation.receive_audio(chunk)
    except KeyboardInterrupt:
        print("\nReceived interrupt signal...")
    except Exception as e:
        print(f"\n[ERROR] Error in main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nTerminating conversation...")
        await conversation.terminate()
        print("Conversation ended. Goodbye!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")