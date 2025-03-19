from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
from datetime import datetime
import threading
import queue
import numpy as np
from src.ml.deep_emotion_model import EEGEmotionClassifier
import traceback
import time
import subprocess
import sys
from pathlib import Path
from src.chatbot.chatbot_service import ChatbotService
import werkzeug.serving

# Add project root to path
project_root = str(Path(__file__).resolve().parent)
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('server.log')
    ]
)

# Create logger
logger = logging.getLogger(__name__)

# Create a filter for logs
class LogFilter(logging.Filter):
    def filter(self, record):
        # Skip the following logs:
        # 1. GET /api/emotion endpoint logs
        # 2. 127.0.0.1 access logs
        # 3. Regular prediction logs
        return not (
            'GET /api/emotion' in str(record.msg) or
            '127.0.0.1' in str(record.msg) or
            ('Prediction:' in str(record.msg) and 'Confidence:' in str(record.msg))
        )

# Apply filter to both werkzeug and our logger
logging.getLogger('werkzeug').addFilter(LogFilter())
logger.addFilter(LogFilter())

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables
eeg_data_queue = queue.Queue(maxsize=1)
emotion_classifier = None
last_prediction = {"emotion": "neutral", "confidence": 0.5, "timestamp": datetime.now()}
prediction_lock = threading.Lock()
PREDICTION_INTERVAL = 5  # seconds
chords_process = None
chatbot = None

def initialize_classifier():
    """Initialize the emotion classifier"""
    global emotion_classifier
    try:
        emotion_classifier = EEGEmotionClassifier()
        logger.info("Emotion classifier initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing classifier: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def initialize_chatbot():
    """Initialize the chatbot service"""
    global chatbot
    try:
        chatbot = ChatbotService()
        logger.info("Chatbot initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing chatbot: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def start_chords_stream():
    """Start the CHORDS stream using chords.py"""
    global chords_process
    try:
        chords_script = Path(project_root) / 'src' / 'chords' / 'chords.py'
        if not chords_script.exists():
            raise FileNotFoundError(f"CHORDS script not found at {chords_script}")
        
        logger.info("Starting CHORDS stream...")
        chords_process = subprocess.Popen([sys.executable, str(chords_script), '--lsl'], 
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
        logger.info("CHORDS stream started successfully")
        
    except Exception as e:
        logger.error(f"Error starting CHORDS stream: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def prediction_worker():
    """Worker function to make predictions exactly every 5 seconds"""
    global last_prediction
    logger.info("Starting prediction worker thread")
    
    while True:
        try:
            # Get the current time
            current_time = datetime.now()
            
            try:
                # Get latest EEG data with buffer
                eeg_data = eeg_data_queue.get_nowait()
                
                # First row is the latest sample, rest is the buffer
                latest_sample = eeg_data[0]
                buffer_data = eeg_data[1:]
                
                # Make prediction using both latest sample and buffer
                with prediction_lock:
                    emotion, confidence = emotion_classifier.predict_realtime(buffer_data)  # Pass full buffer for FFT
                    current_time = datetime.now()
                    
                    # Try to start conversation if chatbot exists and setup is complete
                    chat_message = None
                    if chatbot is not None and chatbot.setup_complete and chatbot.has_started_conversation:
                        try:
                            # Only get response if emotion changes or significant events
                            if (emotion != last_prediction.get("emotion") or 
                                abs(confidence - last_prediction.get("confidence", 0)) > 0.2):
                                initial_response = chatbot.get_response(current_emotion=emotion, confidence=confidence)
                                if initial_response:
                                    logger.info("Chatbot generated new response")
                                    chat_message = initial_response
                        except Exception as chat_e:
                            logger.error(f"Error in chatbot response: {str(chat_e)}")
                    
                    last_prediction = {
                        "emotion": emotion,
                        "confidence": confidence,
                        "timestamp": current_time,
                        "eeg_data": latest_sample.tolist(),  # Only send latest sample to frontend
                        "chat_message": chat_message  # Include any chat messages
                    }
                    logger.info(f"Emotion: {emotion}, Confidence: {confidence:.2f}")
                    
            except queue.Empty:
                pass
                
            # Calculate time to next 5-second mark using timestamp
            current_timestamp = time.time()
            next_prediction = current_timestamp - (current_timestamp % PREDICTION_INTERVAL) + PREDICTION_INTERVAL
            sleep_time = max(0, next_prediction - current_timestamp)
            time.sleep(sleep_time)
                
        except Exception as e:
            logger.error(f"Error in prediction worker: {str(e)}")
            logger.error(traceback.format_exc())  # Add full traceback for debugging
            time.sleep(5)

def eeg_stream_worker():
    """Worker function for EEG stream"""
    try:
        logger.info("Looking for CHORDS stream...")
        # Wait for CHORDS stream to start
        time.sleep(2)
        
        from pylsl import StreamInlet, resolve_streams
        streams = resolve_streams()  # Look specifically for CHORDS stream
        if not streams:
            raise RuntimeError("No CHORDS stream found")
            
        inlet = StreamInlet(streams[0])
        logger.info("Connected to CHORDS stream")
        
        # Initialize data buffer for FFT
        buffer_size = 256  # Increased buffer size for better FFT resolution
        data_buffer = np.zeros((buffer_size, 2), dtype=np.float32)
        buffer_index = 0
        
        # Pull multiple samples at once for efficiency
        chunk_size = 32  # Process 32 samples at a time
        while True:
            # Get multiple samples at once
            samples, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=chunk_size)
            if samples:
                try:
                    # Convert samples to numpy array
                    chunk_data = np.array(samples, dtype=np.float32)
                    
                    # Add new samples to buffer
                    samples_to_add = min(len(chunk_data), buffer_size - buffer_index)
                    data_buffer[buffer_index:buffer_index + samples_to_add] = chunk_data[:samples_to_add]
                    buffer_index += samples_to_add
                    
                    # If buffer is full, process it
                    if buffer_index >= buffer_size:
                        # Take the most recent sample for prediction
                        latest_sample = chunk_data[-1]  # Get the last sample from the chunk
                        eeg_data = np.array([[float(latest_sample[0]), float(latest_sample[1])]], dtype=np.float32)
                        
                        # Update queue with latest data and full buffer
                        if eeg_data_queue.full():
                            eeg_data_queue.get_nowait()  # Remove old data
                        eeg_data_queue.put_nowait(np.vstack((eeg_data, data_buffer)))
                        
                        # Reset buffer
                        buffer_index = 0
                        data_buffer.fill(0)
                    
                except (ValueError, queue.Full) as e:
                    continue
                    
    except Exception as e:
        logger.error(f"Error in EEG stream: {str(e)}")
        logger.error(traceback.format_exc())  # Add full traceback for better debugging

@app.route('/api/emotion')
def get_emotion():
    """API endpoint to get current emotion and EEG data"""
    try:
        # Ensure classifier is initialized
        if emotion_classifier is None:
            initialize_classifier()
        
        # Return the last prediction without making a new one
        with prediction_lock:
            current_time = datetime.now()
            time_since_last = (current_time - last_prediction["timestamp"]).total_seconds()
            
            response = {
                'emotion': last_prediction["emotion"],
                'confidence': last_prediction["confidence"],
                'relaxed_confidence': 1 - last_prediction["confidence"] if last_prediction["emotion"] == "stressed" else last_prediction["confidence"],
                'stressed_confidence': last_prediction["confidence"] if last_prediction["emotion"] == "stressed" else 1 - last_prediction["confidence"],
                'eeg_data': last_prediction.get("eeg_data", [0.0, 0.0]),
                'timestamp': last_prediction["timestamp"].isoformat(),
                'next_prediction_in': max(0, PREDICTION_INTERVAL - time_since_last),
                'chat_message': last_prediction.get("chat_message"),  # Include chat message if available
                'is_setup_phase': chatbot.is_first_emotion if chatbot else True,
                'setup_complete': chatbot.setup_complete if chatbot else False
            }
            
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        return jsonify({
            'error': str(e),
            'emotion': 'neutral',
            'confidence': 0.5,
            'eeg_data': [0.0, 0.0],
            'next_prediction_in': PREDICTION_INTERVAL,
            'chat_message': None,
            'is_setup_phase': True,
            'setup_complete': False
        })

@app.route('/api/chat', methods=['POST'])
def chat():
    """API endpoint for chatbot interaction"""
    try:
        # Ensure chatbot is initialized
        if chatbot is None:
            initialize_chatbot()
        
        # Get message from request
        data = request.get_json()
        user_message = data.get('message', '')
        
        # Get current emotion state
        with prediction_lock:
            current_emotion = last_prediction["emotion"]
            confidence = last_prediction["confidence"]
        
        # Get chatbot response
        response = chatbot.get_response(user_message, current_emotion, confidence)
        
        # Return response in the correct format
        return jsonify({
            'response': response,
            'emotion': current_emotion,
            'confidence': confidence,
            'is_setup_phase': not chatbot.has_started_conversation,
            'setup_complete': chatbot.setup_complete
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'response': "I apologize, but I'm having trouble processing your request.",
            'is_setup_phase': True,
            'setup_complete': False
        }), 500

@app.route('/api/chat/clear', methods=['POST'])
def clear_chat():
    """API endpoint to clear chat history"""
    try:
        if chatbot is not None:
            chatbot.clear_history()
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Error clearing chat history: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Add a new endpoint to get initial message
@app.route('/api/chat/initial', methods=['GET'])
def get_initial_message():
    """API endpoint to get the initial setup message"""
    try:
        # Ensure chatbot is initialized
        if chatbot is None:
            initialize_chatbot()
        
        # Get current emotion state
        with prediction_lock:
            current_emotion = last_prediction["emotion"]
            confidence = last_prediction["confidence"]
        
        # Get appropriate message based on state
        if not chatbot.setup_complete:
            response = chatbot.get_setup_message()
        elif not chatbot.has_started_conversation and current_emotion != "neutral":
            response = chatbot.start_conversation(current_emotion, confidence)
        else:
            response = chatbot.get_setup_message()
        
        return jsonify({
            'response': response,
            'emotion': current_emotion,
            'confidence': confidence,
            'is_setup_phase': not chatbot.has_started_conversation,
            'setup_complete': chatbot.setup_complete
        })
        
    except Exception as e:
        logger.error(f"Error getting initial message: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'response': "Hello! Please wait while I initialize...",
            'is_setup_phase': True,
            'setup_complete': False
        }), 500

def cleanup():
    """Cleanup function to terminate CHORDS process"""
    global chords_process
    if chords_process:
        logger.info("Terminating CHORDS process...")
        chords_process.terminate()
        chords_process.wait()
        logger.info("CHORDS process terminated")

if __name__ == '__main__':
    try:
        # Initialize classifier
        initialize_classifier()
        
        # Initialize chatbot
        initialize_chatbot()
        
        # Start CHORDS stream
        start_chords_stream()
        
        # Start EEG stream in background thread
        eeg_thread = threading.Thread(target=eeg_stream_worker, daemon=True)
        eeg_thread.start()
        
        # Start prediction worker in background thread
        prediction_thread = threading.Thread(target=prediction_worker, daemon=True)
        prediction_thread.start()
        
        # Run Flask server
        app.run(host='0.0.0.0', port=5000)
        
    finally:
        cleanup() 