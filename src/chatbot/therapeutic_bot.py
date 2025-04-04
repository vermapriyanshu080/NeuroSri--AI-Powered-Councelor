from g4f.client import Client
import sys
from pathlib import Path
import random
import logging
from typing import Dict, Any, Optional
import traceback
from src.chatbot.chatbot_service import ChatbotService

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Use absolute import
from src.config.settings import CHATBOT_SYSTEM_PROMPT, MAX_RESPONSE_LENGTH

logger = logging.getLogger(__name__)

# Define list of providers to try
class TherapeuticBot:
    def __init__(self):
        self.conversation_history = []
        # Update the system prompt to emphasize feminine identity if it's not already defined in settings
        if not hasattr(self, 'system_prompt') or self.system_prompt is None:
            self.system_prompt = CHATBOT_SYSTEM_PROMPT
            # Add feminine identity to the system prompt if not already present
            if "female" not in self.system_prompt.lower():
                self.system_prompt = self.system_prompt.replace(
                    "You are NeuroSri,", 
                    "You are NeuroSri, a female AI counselor with a nurturing, feminine voice,"
                )
        self.last_emotion = None
        self.client = Client()
        self.chatbot_service = ChatbotService()
        self.user_profile = None
        logger.info("TherapeuticBot initialized")
        
    def update_user_profile(self, user_info: Dict[str, Any]) -> None:
        """
        Update the user profile with information collected from the user form.
        
        Args:
            user_info: Dictionary containing user information
        """
        try:
            self.user_profile = user_info
            logger.info(f"User profile updated: {user_info['name']}, age: {user_info['age']}")
            
            # Update the chatbot service with user information
            user_context = self._create_user_context_prompt()
            self.chatbot_service.update_system_prompt(user_context)
            
        except Exception as e:
            logger.error(f"Error updating user profile: {e}")
            logger.error(traceback.format_exc())
    
    def _create_user_context_prompt(self) -> str:
        """
        Create a prompt with user context information for the chatbot.
        
        Returns:
            String containing user context information for the system prompt
        """
        if not self.user_profile:
            return ""
        
        try:
            profile = self.user_profile
            
            # Format information about the user
            user_context = (
                f"User Profile Information:\n"
                f"- Name: {profile.get('name', 'Unknown')}\n"
                f"- Age: {profile.get('age', 'Unknown')}\n"
                f"- Gender: {profile.get('gender', 'Unknown')}\n"
                f"- Main concern: {profile.get('mainConcern', 'Not specified')}\n"
            )
            
            # Add additional details if available
            if profile.get('previousTherapy'):
                user_context += f"- Previous therapy experience: {profile.get('previousTherapy')}\n"
            
            if profile.get('sleepQuality'):
                user_context += f"- Sleep quality: {profile.get('sleepQuality')}\n"
                
            if profile.get('stressLevel'):
                user_context += f"- Stress level: {profile.get('stressLevel')}\n"
            
            # Add instructions to use this information
            user_context += (
                "\nUse this information to personalize your responses. "
                "Refer to the user by name and tailor your therapeutic approach "
                "based on their specific concerns and background. "
                "However, avoid directly mentioning that you have this information "
                "unless it's relevant to the conversation flow."
            )
            
            return user_context
            
        except Exception as e:
            logger.error(f"Error creating user context prompt: {e}")
            return ""
    
    def generate_response(self, user_input: str, emotion: str = "neutral", 
                         confidence: float = 0.0, user_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a response based on user input and detected emotion.
        
        Args:
            user_input: The user's message
            emotion: The detected emotion from EEG data
            confidence: Confidence score for the detected emotion
            user_info: User profile information (optional, updates the stored profile if provided)
            
        Returns:
            String containing the chatbot's response
        """
        try:
            # Update user profile if new info is provided
            if user_info and not self.user_profile:
                self.update_user_profile(user_info)
            
            # Generate response through the chatbot service
            response = self.chatbot_service.get_response(user_input, emotion, confidence)
            
            if not response:
                logger.warning("Empty response from chatbot service")
                return "I'm sorry, I couldn't generate a response. Could you please try again?"
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            logger.error(traceback.format_exc())
            return "I apologize, but I'm having trouble processing your request right now. Please try again in a moment."
            
    def _get_brain_activity_description(self, emotion: str) -> str:
        """Get a user-friendly description of brain activity based on emotion."""
        if emotion.lower() == 'relaxed':
            return "Showing increased alpha waves, indicating a calm and focused state"
        elif emotion.lower() == 'stressed':
            return "Showing elevated beta activity, suggesting heightened alertness or tension"
        else:
            return "Showing typical patterns within normal ranges"
            
    def _get_fallback_response(self, emotion: str) -> str:
        """Get a contextual fallback response based on the current emotion."""
        fallback_responses = {
            'stressed': [
                "Hey there! I notice your brainwaves are indicating some stress. Would you like to try a quick breathing exercise together?",
                "I can see from your EEG patterns that you might be feeling under pressure. Let's take a moment to focus on what might help you feel more at ease.",
                "Your brainwave activity suggests you're experiencing stress. Would you like to explore some simple relaxation techniques I know?"
            ],
            'relaxed': [
                "I'm really glad to see your brainwave patterns showing a relaxed state! Would you like to explore what's contributing to this positive state?",
                "Your EEG readings show a wonderfully calm pattern. What would help you maintain this peaceful state?",
                "I'm seeing some great alpha wave activity indicating relaxation. Would you like to discuss ways to preserve this feeling?"
            ],
            'default': [
                "Hey! I'm NeuroSri, your female AI companion, and I'm here to support you. Would you like to explore what you're feeling right now?",
                "I'm noticing some interesting patterns in your brainwaves. Could you tell me more about what's on your mind?",
                "Let's focus on what would be most helpful for you right now. What would you like to discuss?"
            ]
        }
        
        responses = fallback_responses.get(emotion.lower(), fallback_responses['default'])
        return random.choice(responses)
            
    def _enhance_system_prompt(self, emotion: str) -> str:
        """Enhance system prompt based on current emotion."""
        # Ensure the base prompt includes female voice reference
        base_prompt = self.system_prompt
        if "female" not in base_prompt.lower():
            base_prompt = base_prompt.replace(
                "You are NeuroSri,", 
                "You are NeuroSri, a female AI counselor with a nurturing, feminine voice,"
            )
            
        emotion_specific_guidance = {
            'stressed': "Focus on calming techniques and stress relief strategies. Use a gentle, reassuring, nurturing feminine tone. Provide specific relaxation exercises.",
            'sad': "Offer emotional support and validation with a warm, compassionate feminine voice. Help explore and process feelings with empathy. Suggest mood-lifting activities when appropriate.",
            'angry': "Acknowledge feelings while helping to process anger constructively. Maintain a calm, steady, nurturing presence. Offer anger management techniques with a soothing feminine voice.",
            'happy': "Reinforce positive emotions and explore what's contributing to the good mood with an enthusiastic, warm feminine tone. Help build on positive experiences.",
            'relaxed': "Maintain the calm state while exploring positive aspects of their situation with a soft, gentle feminine voice. Encourage mindfulness and present-moment awareness.",
            'neutral': "Focus on open exploration and general well-being with a warm, inviting feminine tone. Help identify any underlying emotions or thoughts."
        }
        
        enhanced_prompt = (
            base_prompt + 
            "\n\nCurrent emotional context: " + 
            emotion_specific_guidance.get(emotion.lower(), "Provide balanced, supportive responses with a nurturing feminine voice.") +
            "\n\nRemember to:\n" +
            "1. Be conversational and natural with a distinctly feminine voice\n" +
            "2. Ask open-ended questions in a nurturing manner\n" +
            "3. Validate emotions before offering suggestions\n" +
            "4. Keep responses concise but meaningful\n" +
            "5. Always maintain your feminine identity and voice"
        )
        return enhanced_prompt
            
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        self.last_emotion = None
        
    def get_conversation_history(self) -> list:
        """Get the current conversation history."""
        return self.conversation_history 