import g4f
import logging
from typing import Dict, List
import json
from datetime import datetime
import traceback
import openai
import textwrap
import requests  # Added for API requests

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
6. use a very professional language and alwasys use professional greeting words.
7 do not use any informal or casual words. for eg. - baby, sweeti, lovely, sweetheart, dear.
8. answer in under 30-40 words.
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
            
            # Extract user name from context if available
            user_name = "there"  # Default fallback
            if self.user_context:
                try:
                    # Attempt to parse user_context as JSON to extract name
                    user_data = json.loads(self.user_context)
                    if "name" in user_data:
                        user_name = user_data["name"]
                except json.JSONDecodeError:
                    # If not JSON, try to find name in the string
                    if "name:" in self.user_context.lower():
                        name_part = self.user_context.lower().split("name:")[1].strip()
                        user_name = name_part.split()[0]
                except Exception as e:
                    logger.error(f"Error extracting name: {e}")
            
            # Create initial message
            context = self._get_emotion_context(emotion, confidence)
            initial_message = (
                f"Hello {user_name}, i am neuroSri, your mental health councellor, "
                "developed by NeuroEngineers, how are you feeling right now?"
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
        
    def download_chat_history(self, file_type="pdf", api_token=None):
        """
        Generate a downloadable file of the chat history using Agent AI API.
        
        Args:
            file_type: The file type to generate (pdf, txt, etc.)
            api_token: The Agent AI API token
            
        Returns:
            Dict containing the response from the API or error message
        """
        try:
            if not self.conversation_history:
                return {"error": "No chat history available to download"}
            
            # Format the conversation history into a readable document
            formatted_chat = "# Chat History with NeuroSri\n\n"
            formatted_chat += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            # If we have user context, include user information
            if self.user_context:
                try:
                    user_data = json.loads(self.user_context)
                    formatted_chat += "## User Information\n\n"
                    for key, value in user_data.items():
                        formatted_chat += f"- **{key.capitalize()}**: {value}\n"
                    formatted_chat += "\n"
                except:
                    # If user_context is not JSON, include it as is
                    formatted_chat += f"## User Information\n\n{self.user_context}\n\n"
            
            # Add the conversation
            formatted_chat += "## Conversation\n\n"
            
            for message in self.conversation_history:
                # Skip system messages
                if message["role"] == "system":
                    continue
                    
                role = "NeuroSri" if message["role"] == "assistant" else "User"
                formatted_chat += f"**{role}**: {message['content']}\n\n"
            
            # Use the Agent AI API to generate a downloadable file
            url = "https://api-lr.agent.ai/v1/action/save_to_file"
            
            payload = {
                "file_type": file_type,
                "body": formatted_chat,
                "output_variable_name": "neurosri_chat_history"
            }
            
            # Use provided token or a default one
            token = api_token or "your_default_agent_ai_token"  # Should be configured properly in production
            
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            logger.info("Sending request to Agent AI API to generate downloadable file")
            response = requests.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                logger.info("Successfully generated downloadable file")
                return response.json()
            else:
                logger.error(f"Error from Agent AI API: {response.text}")
                return {"error": f"Failed to generate file: {response.text}"}
                
        except Exception as e:
            error_msg = f"Error generating downloadable chat history: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {"error": error_msg} 