from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import sys
from pathlib import Path
import numpy as np
import g4f  # This suggests using GPT4Free for responses
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Now use absolute imports
from src.chatbot.therapeutic_bot import TherapeuticBot
from src.eeg.data_acquisition import EEGStream
from src.eeg.signal_processing import SignalProcessor
from src.ml.deep_emotion_model import DeepEmotionClassifier

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Enable CORS for all domains
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Global variable to store user information
user_info = None

# Initialize components with error handling
try:
    logger.info("Initializing EEG stream connection...")
    eeg_stream = EEGStream()
    signal_processor = SignalProcessor()
    
    # Initialize emotion classifier with model download if needed
    logger.info("Loading emotion classifier model...")
    emotion_classifier = DeepEmotionClassifier()
    
    chatbot = TherapeuticBot()
    logger.info("All components initialized successfully")
except Exception as e:
    logger.error(f"Error initializing components: {e}")
    sys.exit(1)

@app.route('/api/emotion', methods=['GET'])
def get_emotion():
    try:
        # Get latest EEG data
        result = eeg_stream.get_data()
        if result is not None:
            eeg_data, timestamp = result
            
            # Ensure data is numpy array with correct dtype and shape
            eeg_data = np.asarray(eeg_data, dtype=np.float32)
            
            # Reshape based on input shape
            if len(eeg_data.shape) == 1:
                eeg_data = eeg_data.reshape(-1, 2)  # Reshape to (samples, channels)
            elif len(eeg_data.shape) == 2 and eeg_data.shape[1] != 2:
                eeg_data = eeg_data.T  # Transpose if channels are in rows
            
            # Get current time
            current_time = datetime.now()
            
            # Get the latest prediction from the emotion classifier
            emotion, confidence = emotion_classifier.predict_realtime(eeg_data)
            logger.info(f"Emotion: {emotion}, Confidence: {confidence:.2f}")
            
            # Calculate time until next 5-second mark
            seconds_until_next = 5 - (current_time.timestamp() % 5)
            
            return jsonify({
                'emotion': emotion,
                'confidence': confidence,
                'eeg_data': eeg_data[:, 0].tolist(),  # First channel data
                'timestamp': current_time.isoformat(),
                'next_prediction_in': seconds_until_next  # Dynamic time until next prediction
            })
        
        return jsonify({
            'emotion': 'neutral',
            'confidence': 0,
            'eeg_data': [],
            'timestamp': datetime.now().isoformat(),
            'next_prediction_in': 5
        })
    except Exception as e:
        logger.error(f"Error in emotion endpoint: {e}")
        return jsonify({'error': str(e)}), 500

def get_current_eeg_analysis():
    try:
        # Get latest EEG data
        result = eeg_stream.get_data()
        if result is not None:
            eeg_data, timestamp = result
            
            # Ensure data is numpy array with correct dtype and shape
            eeg_data = np.asarray(eeg_data, dtype=np.float32)
            
            # Reshape based on input shape
            if len(eeg_data.shape) == 1:
                eeg_data = eeg_data.reshape(-1, 2)  # Reshape to (samples, channels)
            elif len(eeg_data.shape) == 2 and eeg_data.shape[1] != 2:
                eeg_data = eeg_data.T  # Transpose if channels are in rows
            
            # Process the signal
            features = signal_processor.preprocess_data(eeg_data)
            if features is not None:
                # Force a new prediction each time
                emotion, confidence = emotion_classifier.predict_realtime(eeg_data)
                logger.info(f"Emotion: {emotion}, Confidence: {confidence:.2f}")
                
                # Calculate wave powers
                alpha_power = signal_processor.get_band_power(eeg_data[:, 0], 'alpha')
                beta_power = signal_processor.get_band_power(eeg_data[:, 0], 'beta')
                theta_power = signal_processor.get_band_power(eeg_data[:, 0], 'theta')
                delta_power = signal_processor.get_band_power(eeg_data[:, 0], 'delta')
                
                return {
                    'emotion': emotion,
                    'confidence': confidence,
                    'alpha_power': float(alpha_power),
                    'beta_power': float(beta_power),
                    'theta_delta_ratio': float(theta_power / delta_power if delta_power > 0 else 0),
                    'eeg_data': eeg_data[-100:, 0].tolist(),  # Last 100 samples of first channel
                    'timestamp': datetime.now().isoformat()
                }
        
        return {
            'emotion': 'neutral',
            'confidence': 0,
            'alpha_power': 0,
            'beta_power': 0,
            'theta_delta_ratio': 0,
            'eeg_data': [],
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting EEG analysis: {e}")
        return None

@app.route('/api/user-info', methods=['POST'])
def submit_user_info():
    global user_info
    try:
        data = request.get_json()
        
        if not data or 'userInfo' not in data:
            return jsonify({
                'success': False,
                'error': 'No user information provided'
            }), 400
        
        # Store user info in global variable
        user_info = data['userInfo']
        
        # Log the received user information
        logger.info(f"Received user information: {user_info}")
        
        # Update the chatbot with user information
        if chatbot:
            try:
                # Add user profile info to chatbot context
                chatbot.update_user_profile(user_info)
                logger.info("Chatbot updated with user information")
            except Exception as e:
                logger.error(f"Error updating chatbot with user info: {e}")
                # Continue anyway as this is not critical
        
        return jsonify({
            'success': True,
            'message': 'User information saved successfully'
        })
    except Exception as e:
        logger.error(f"Error in user-info endpoint: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'status': 'error',
                'error': 'No message provided'
            }), 400
        
        message = data['message']
        # Include user info if available
        user_data = user_info if user_info else data.get('userInfo')
        
        # Get current emotion analysis from EEG data
        emotion_result = get_current_eeg_analysis()
        emotion_data = emotion_result.get_json() if hasattr(emotion_result, 'get_json') else {}
        
        current_emotion = emotion_data.get('emotion', 'neutral')
        confidence = emotion_data.get('confidence', 0)
        eeg_context = emotion_data.get('eeg_context', 'EEG signal is not available or inconclusive.')
        
        logger.info(f"Chat request with message: '{message}'")
        logger.info(f"Current emotion: {current_emotion}, Confidence: {confidence}")
        
        # Generate chatbot response, including user info if available
        response = chatbot.generate_response(message, current_emotion, confidence, user_data)
        
        return jsonify({
            "response": response,
            "emotion": current_emotion,
            "confidence": confidence,
            "status": "success",
            "eeg_context": eeg_context,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        return jsonify({
            "error": str(e),
            "status": "error",
            "fallback_response": "I apologize, but I'm having trouble processing your message right now. Could you please try again?"
        }), 500

# Replace before_first_request with a better initialization approach
def initialize_streaming():
    global eeg_stream
    logger.info("Attempting to connect to LSL stream...")
    if eeg_stream.connect():
        logger.info("✓ Flask server successfully connected to LSL stream")
    else:
        logger.error("✗ Failed to connect to LSL stream - Please ensure LSL stream is running")

# Initialize streaming when the app starts
with app.app_context():
    initialize_streaming()

if __name__ == '__main__':
    try:
        print("Starting Flask server...")
        # Allow connections from any IP
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        print(f"Error starting server: {e}") 