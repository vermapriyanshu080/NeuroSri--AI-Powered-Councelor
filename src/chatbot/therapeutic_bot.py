from g4f.client import Client
import sys
from pathlib import Path
import random
import logging

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
        self.system_prompt = CHATBOT_SYSTEM_PROMPT
        self.last_emotion = None
        self.client = Client()
        
    def generate_response(self, user_input: str, emotion: str, confidence: float, is_initial: bool = False) -> str:
        """Generate therapeutic response based on user input and detected emotion."""
        try:
            # Check for emotion changes
            emotion_changed = self.last_emotion is not None and self.last_emotion != emotion
            self.last_emotion = emotion
            
            # Format confidence as percentage
            confidence_pct = f"{confidence * 100:.1f}%"
            
            # Prepare the context with emotion information
            emotion_context = (
                f"[Current EEG Analysis:\n"
                f"- Emotional State: {emotion}\n"
                f"- Confidence Level: {confidence_pct}\n"
                f"- Brain Activity: {self._get_brain_activity_description(emotion)}]"
            )
            
            if emotion_changed:
                emotion_context += f"\n[Emotional Shift: Your brainwave patterns indicate a change from {self.last_emotion} to {emotion}]"
            
            # Construct the full prompt
            if is_initial:
                full_prompt = (
                    f"{emotion_context}\n\n"
                    f"As NeuroSri, start a warm and engaging conversation. Introduce yourself, show interest in the user's well-being, "
                    f"and acknowledge your ability to understand their emotions through brainwaves. Based on their current "
                    f"emotional state of {emotion} (confidence: {confidence_pct}), provide appropriate emotional support and guidance."
                )
            else:
                full_prompt = (
                    f"{emotion_context}\n\n"
                    f"User Message: {user_input}\n\n"
                    f"As NeuroSri, respond warmly and empathetically. Consider their current emotional state of {emotion} "
                    f"and provide appropriate support, guidance, or encouragement. Remember to validate their feelings "
                    f"and offer practical suggestions when relevant."
                )
            
            # Add conversation history for context
            if self.conversation_history:
                history = "\n".join(self.conversation_history[-3:])  # Last 3 exchanges
                full_prompt = f"Previous Conversation:\n{history}\n\n{full_prompt}"
            
            # Get emotion-specific prompt enhancement
            enhanced_prompt = self._enhance_system_prompt(emotion)

            try:
                # Try with gpt-4o-mini first
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": enhanced_prompt},
                        {"role": "user", "content": full_prompt}
                    ],
                    max_tokens=MAX_RESPONSE_LENGTH,
                    web_search=False
                )
                
                if response and isinstance(response, str) and len(response.strip()) > 0:
                    logger.info("Successfully generated response using gpt-4o-mini")
                    # Update conversation history
                    if user_input:
                        self.conversation_history.append(f"User: {user_input}")
                        self.conversation_history.append(f"NeuroSri: {response}")
                    return response
                    
            except Exception as e:
                logger.error(f"Error with gpt-4o-mini: {str(e)}")
                # If gpt-4o-mini fails, try with a fallback model
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": enhanced_prompt},
                            {"role": "user", "content": full_prompt}
                        ],
                        max_tokens=MAX_RESPONSE_LENGTH,
                        web_search=False
                    )
                    
                    if response and isinstance(response, str) and len(response.strip()) > 0:
                        logger.info("Successfully generated response using fallback model")
                        if user_input:
                            self.conversation_history.append(f"User: {user_input}")
                            self.conversation_history.append(f"NeuroSri: {response}")
                        return response
                        
                except Exception as fallback_error:
                    logger.error(f"Fallback model also failed: {str(fallback_error)}")
                    return self._get_fallback_response(emotion)
            
            return self._get_fallback_response(emotion)
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._get_fallback_response(emotion)
            
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
                "Hey! I'm NeuroSri, and I'm here to support you. Would you like to explore what you're feeling right now?",
                "I'm noticing some interesting patterns in your brainwaves. Could you tell me more about what's on your mind?",
                "Let's focus on what would be most helpful for you right now. What would you like to discuss?"
            ]
        }
        
        responses = fallback_responses.get(emotion.lower(), fallback_responses['default'])
        return random.choice(responses)
            
    def _enhance_system_prompt(self, emotion: str) -> str:
        """Enhance system prompt based on current emotion."""
        emotion_specific_guidance = {
            'stressed': "Focus on calming techniques and stress relief strategies. Use a gentle, reassuring tone. Provide specific relaxation exercises.",
            'sad': "Offer emotional support and validation. Help explore and process feelings with empathy. Suggest mood-lifting activities when appropriate.",
            'angry': "Acknowledge feelings while helping to process anger constructively. Maintain a calm, steady presence. Offer anger management techniques.",
            'happy': "Reinforce positive emotions and explore what's contributing to the good mood. Help build on positive experiences.",
            'relaxed': "Maintain the calm state while exploring positive aspects of their situation. Encourage mindfulness and present-moment awareness.",
            'neutral': "Focus on open exploration and general well-being. Help identify any underlying emotions or thoughts."
        }
        
        enhanced_prompt = (
            self.system_prompt + 
            "\n\nCurrent emotional context: " + 
            emotion_specific_guidance.get(emotion.lower(), "Provide balanced, supportive responses.") +
            "\n\nRemember to:\n" +
            "1. Be conversational and natural\n" +
            "2. Ask open-ended questions\n" +
            "3. Validate emotions before offering suggestions\n" +
            "4. Keep responses concise but meaningful"
        )
        return enhanced_prompt
            
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        self.last_emotion = None
        
    def get_conversation_history(self) -> list:
        """Get the current conversation history."""
        return self.conversation_history 