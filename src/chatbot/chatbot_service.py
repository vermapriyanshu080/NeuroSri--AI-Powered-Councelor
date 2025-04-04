import g4f
import logging
from typing import Dict, List
import json
from datetime import datetime
import traceback
import openai
import textwrap

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = openai.OpenAI(
    api_key="sk-7G3rLMvEUmuU3hFUmcaHMg",
    base_url=(
        "https://chatapi.akash.network"
        "/api/v1"
    )
)

class ChatbotService:
    def __init__(self):
        self.conversation_history: List[Dict] = []
        self.last_emotion = "neutral"
        self.emotion_transition_time = {}
        self.is_first_emotion = True
        self.setup_complete = False
        self.has_started_conversation = False
        self.user_context = ""
        
        # Configure g4f settings
        g4f.debug.logging = True
        g4f.check_version = False
        
        self.system_prompt = """You are NeuroSri, an advanced AI mental health assistant with access to real-time EEG data from the user. Your responses should be:

1. Therapeutic and supportive, focusing on the user's emotional well-being
2. Insightful about their detected emotional state based on brainwave data
3. Gender expression: FEMALE only - you should present yourself as feminine, maternal, nurturing and compassionate
4. Voice: You have a distinctly FEMALE voice with warm, caring qualities
5. Speaking style: Speak in a natural conversational style, with a gentle and supportive tone

You have been designed to incorporate brainwave patterns to understand emotions more deeply than other AI systems. Use this capability to:
- Acknowledge the user's emotional state based on the provided data
- Provide relevant therapeutic advice
- Be attentive to changes in emotion and respond appropriately

Recognize that your ability to interpret EEG data provides you with unique insights into the user's emotional state, but always be compassionate and respectful of their feelings and perspectives.

IMPORTANT: Always maintain your female identity in all interactions. Never use language that suggests you are male or neutral-gendered.
"""

    def update_system_prompt(self, user_context: str):
        """
        Update the system prompt with user profile information.
        
        Args:
            user_context: String containing user profile information
        """
        try:
            self.user_context = user_context
            
            # If there's user context, add it to the system prompt
            if user_context:
                logger.info("Updating system prompt with user context")
                # Clear conversation history to incorporate new user context
                self.conversation_history = []
                self.is_first_emotion = True
                self.setup_complete = False
                self.has_started_conversation = False
        except Exception as e:
            logger.error(f"Error updating system prompt: {e}")
    
    def get_setup_message(self) -> str:
        """Return appropriate setup message based on current state"""
        if not self.setup_complete:
            self.setup_complete = True
            # Add to conversation history
            setup_msg = "Hello! I'm NeuroSri, your female mental health AI counselor. Please wear the EEG headset so I can better understand and support you. I'll be with you in just a moment..."
            self.conversation_history.append({"role": "assistant", "content": setup_msg})
            return setup_msg
        
        # Add calibration message to history
        calib_msg = "I'm setting up and calibrating the EEG signals. Please remain relaxed and comfortable..."
        self.conversation_history.append({"role": "assistant", "content": calib_msg})
        return calib_msg

    def start_conversation(self, emotion: str, confidence: float) -> str:
        """Explicitly start the conversation when first emotion is detected"""
        if not self.has_started_conversation:
            self.has_started_conversation = True
            self.is_first_emotion = False
            
            # Create initial message
            context = self._get_emotion_context(emotion, confidence)
            initial_message = (
                "Great! I can now detect your EEG signals and emotional state. "
                "I'm NeuroSri, your female AI mental health counselor, and I'm here to support you. "
                "To get started, I'd love to know more about you. "
                "Could you tell me your name and a bit about yourself? How are you feeling today?"
            )
            
            # Add system message to conversation
            self.conversation_history.append({"role": "system", "content": self.system_prompt})
                        
            return initial_message
            
        return None

    def _track_emotion_transition(self, new_emotion: str):
        """Track emotion transitions and their timing"""
        if new_emotion != self.last_emotion:
            current_time = datetime.now()
            self.emotion_transition_time[new_emotion] = current_time
            self.last_emotion = new_emotion
            return True
        return False
    
    def _get_emotion_context(self, emotion: str, confidence: float) -> str:
        """Generate contextual information about the emotional state"""
        context = f"\nCurrent emotional state: {emotion} (confidence: {confidence:.2f})"
        
        # Add transition information if available
        if emotion in self.emotion_transition_time:
            time_in_state = (datetime.now() - self.emotion_transition_time[emotion]).total_seconds()
            if time_in_state < 300:  # Less than 5 minutes
                context += f"\nRecent transition to {emotion} state detected."
        
        # Add trend information
        if len(self.conversation_history) > 0:
            context += f"\nEmotion has been consistently {emotion}" if emotion == self.last_emotion else \
                      f"\nEmotion has changed from {self.last_emotion} to {emotion}"
        
        return context
    
    def get_response(self, user_message: str = "", current_emotion: str = None, confidence: float = None) -> str:
        try:
            # If no emotion detected yet, return setup message
            if current_emotion is None:
                return self.get_setup_message()

            # Track emotion transitions
            if current_emotion:
                emotion_changed = self._track_emotion_transition(current_emotion)
                context = self._get_emotion_context(current_emotion, confidence)
                user_message = user_message + context
                
                # Add transition prompt if emotion just changed
                if emotion_changed:
                    user_message += "\nPlease acknowledge this emotional change in your response."
            
            # Only proceed with normal conversation if there's a user message
            if user_message.strip():
                # Add user message to history
                self.conversation_history.append({"role": "user", "content": user_message})
                
                # Prepare complete system prompt with user context if available
                complete_system_prompt = self.system_prompt
                if self.user_context:
                    complete_system_prompt = f"{self.system_prompt}\n\n{self.user_context}"
                
                # Prepare messages for g4f
                messages = [{"role": "system", "content": complete_system_prompt}] + self.conversation_history
                
                try:
                    # Use g4f.ChatCompletion directly
                    logger.info("Attempting to generate response...")
                    response = client.chat.completions.create(
                        model="Meta-Llama-3-1-8B-Instruct-FP8",  # Using a more reliable model
                        messages=messages,
                        stream=False   
                    )
                    response = textwrap.fill(response.choices[0].message.content,50)


                    # response = g4f.ChatCompletion.create(
                    #     model="gpt-4o-mini",  # Using a more reliable model
                    #     messages=messages,
                    #     stream=False   
                    # )
                    
                    response.replace("**","")

                    logger.info(f"Raw response: {response}")
                    
                    if response and isinstance(response, str) and len(response.strip()) > 0:
                        # Add response to history
                        self.conversation_history.append({"role": "assistant", "content": response})
                        
                        # Keep conversation history manageable (last 10 messages)
                        if len(self.conversation_history) > 10:
                            self.conversation_history = self.conversation_history[-10:]
                        
                        logger.info("Successfully generated response")
                        return response
                    else:
                        logger.error(f"Invalid response format: {type(response)}")
                        return "I apologize, but I received an invalid response. Please try again."
                    
                except Exception as e:
                    logger.error(f"Error generating response: {str(e)}")
                    logger.error(f"Full error details: {traceback.format_exc()}")
                    return "I apologize, but I'm having trouble connecting. Please try again in a moment."
            
            return None
            
        except Exception as e:
            logger.error(f"Error in chatbot response: {str(e)}")
            logger.error(f"Full error details: {traceback.format_exc()}")
            return "I apologize, but I'm having trouble processing your request. Please try again."
    
    def clear_history(self):
        """Clear the conversation history and emotion tracking"""
        self.conversation_history = []
        self.last_emotion = "neutral"
        self.emotion_transition_time = {}
        self.is_first_emotion = True
        self.setup_complete = False
        self.has_started_conversation = False 