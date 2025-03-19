import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path
import logging
import os
from datetime import datetime
import sys
import time
import traceback
from scipy.signal import butter, filtfilt, iirnotch

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, 'eeg_model.log'))
    ]
)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = Path(project_root) / 'models' / 'rf_emotion_model.joblib'
SCALER_PATH = Path(project_root) / 'models' / 'eeg_scaler.joblib'
DATASET_PATH = Path(project_root) / 'dataset' / 'merged_dataset_swapped.csv'
SAMPLING_RATE = 255  # Hz
WINDOW_SIZE = 240    # Window size for feature extraction

# Create models directory if it doesn't exist
os.makedirs(MODEL_PATH.parent, exist_ok=True)

def apply_filters(eeg_data, sampling_rate=255):
    """Apply preprocessing filters to EEG data"""
    try:
        # Notch filter for power line interference (50 Hz)
        notch_freq = 50.0  # For 50 Hz power line
        quality_factor = 30.0  # Quality factor
        b_notch, a_notch = iirnotch(notch_freq, quality_factor, sampling_rate)
        
        # Bandpass filter (0.5 - 45 Hz for EEG)
        nyquist_freq = sampling_rate / 2
        low_cut = 0.5 / nyquist_freq
        high_cut = 45.0 / nyquist_freq
        b_bandpass, a_bandpass = butter(4, [low_cut, high_cut], btype='band')
        
        filtered_data = np.zeros_like(eeg_data)
        
        # Apply filters to each channel
        for channel in range(eeg_data.shape[1]):
            # Apply notch filter
            notched = filtfilt(b_notch, a_notch, eeg_data[:, channel])
            
            # Apply bandpass filter
            filtered_data[:, channel] = filtfilt(b_bandpass, a_bandpass, notched)
            
        return filtered_data
        
    except Exception as e:
        logger.error(f"Error in filter application: {str(e)}")
        logger.error(traceback.format_exc())
        return eeg_data  # Return original data if filtering fails

# Define frequency bands for feature extraction
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)  # Updated to match our filter
}

class EEGEmotionClassifier:
    """Random Forest based EEG emotion classifier"""
    def __init__(self, model_path=None):
        self.model = None
        self.scaler = None
        self.last_prediction = ("neutral", 0.5)  # Store last prediction
        self.window_size = WINDOW_SIZE
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        try:
            if model_path is None:
                model_path = MODEL_PATH
            else:
                model_path = Path(model_path)
            
            if model_path.exists() and SCALER_PATH.exists():
                self.load_model(model_path)
            else:
                logger.info("No existing model found. Training new model...")
                self.train_model()
                
        except Exception as e:
            logger.error(f"Error initializing classifier: {e}")
            raise
    
    def extract_frequency_features(self, eeg_data):
        """Extract frequency band features using FFT with preprocessing filters"""
        try:
            # Ensure contiguous array with proper memory layout and shape
            eeg_data = np.ascontiguousarray(eeg_data, dtype=np.float32)
            
            # Log input shape for debugging
            logger.debug(f"Input EEG data shape: {eeg_data.shape}")
            
            # Reshape if needed - expecting (window_size, 2) shape
            if len(eeg_data.shape) == 3:  # If shape is (batch, window_size, channels)
                eeg_data = eeg_data[0]  # Take first batch
            if eeg_data.shape[1] != 2:  # If channels are not in second dimension
                eeg_data = eeg_data.reshape(-1, 2)
            
            logger.debug(f"Reshaped EEG data shape: {eeg_data.shape}")
            
            # Ensure minimum data points for FFT
            if eeg_data.shape[0] < 4:  # Need minimum points for meaningful FFT
                logger.warning("Not enough data points for FFT, padding with zeros")
                pad_length = max(64, 2 ** int(np.ceil(np.log2(4))))  # Next power of 2, minimum 64
                eeg_data = np.pad(eeg_data, ((0, pad_length - eeg_data.shape[0]), (0, 0)), mode='constant')
            
            # Apply preprocessing filters
            filtered_data = apply_filters(eeg_data, SAMPLING_RATE)
            
            features = []
            
            # Apply window function to reduce spectral leakage
            window = np.hanning(filtered_data.shape[0])[:, np.newaxis]
            windowed_data = filtered_data * window
            
            # Calculate FFT
            fft_data = np.fft.rfft(windowed_data, axis=0)  # Use real FFT
            freqs = np.fft.rfftfreq(filtered_data.shape[0], d=1.0/SAMPLING_RATE)
            
            # Calculate power in each frequency band for each channel
            for channel in range(filtered_data.shape[1]):
                channel_features = []
                power_spectrum = np.abs(fft_data[:, channel]) ** 2
                
                # Extract band powers with safety checks
                for band_name, (low_freq, high_freq) in FREQ_BANDS.items():
                    freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
                    if np.any(freq_mask):  # Check if we have any frequencies in this band
                        band_power = np.mean(power_spectrum[freq_mask])
                    else:
                        logger.warning(f"No frequencies found in {band_name} band, using zero")
                        band_power = 0.0
                    channel_features.append(band_power)
                
                # Calculate statistical features with safety checks
                if len(power_spectrum) > 0:
                    mean_power = np.mean(power_spectrum) if not np.all(np.isnan(power_spectrum)) else 0.0
                    std_power = np.std(power_spectrum) if not np.all(np.isnan(power_spectrum)) else 0.0
                    max_power = np.max(power_spectrum) if not np.all(np.isnan(power_spectrum)) else 0.0
                    min_power = np.min(power_spectrum) if not np.all(np.isnan(power_spectrum)) else 0.0
                else:
                    mean_power, std_power, max_power, min_power = 0.0, 0.0, 0.0, 0.0
                
                channel_features.extend([mean_power, std_power, max_power, min_power])
                features.extend(channel_features)
            
            # Convert to numpy array and ensure correct shape
            features = np.array(features, dtype=np.float32)
            
            # Check for NaN or infinite values
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                logger.warning("Found NaN or infinite values in features, replacing with zeros")
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            logger.debug(f"Extracted features shape: {features.shape}")
            return features
            
        except Exception as e:
            logger.error(f"Error in feature extraction: {str(e)}")
            logger.error(traceback.format_exc())
            return np.zeros(18, dtype=np.float32)  # 9 features per channel * 2 channels
    
    def load_model(self, model_path):
        """Load trained model and scaler"""
        try:
            # Load model and scaler
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(SCALER_PATH)
            logger.info("Model and scaler loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def train_model(self):
        """Train the Random Forest model using cleaned dataset"""
        try:
            # Load and prepare training data
            logger.info("Loading dataset...")
            data = pd.read_csv(DATASET_PATH)
            logger.info(f"Successfully loaded dataset with {len(data)} samples")
            
            # Process training data into windows with FFT features
            X = []
            y = []
            temp_buffer = []
            labels = []
            current_label = None
            
            for idx, row in data.iterrows():
                if idx % 1000 == 0:
                    logger.info(f"Processing row {idx}/{len(data)}")
                
                if current_label is None:
                    current_label = row['Emotion']
                
                if row['Emotion'] == current_label:
                    channel_data = [float(row['Channel 1']), float(row['Channel 2'])]
                    temp_buffer.append(channel_data)
                    labels.append(row['Emotion'])
                    
                    if len(temp_buffer) == self.window_size:
                        # Extract features from window
                        window_array = np.array(temp_buffer, dtype=np.float32)
                        features = self.extract_frequency_features(window_array)
                        X.append(features)
                        y.append(current_label)
                        # Use sliding window with 50% overlap
                        temp_buffer = temp_buffer[self.window_size//2:]
                        labels = labels[self.window_size//2:]
                else:
                    temp_buffer = []
                    labels = []
                    current_label = row['Emotion']
            
            X = np.array(X, dtype=np.float32)
            y = np.array(y)
            
            # Convert labels to numerical format
            label_map = {'Relaxed': 1, 'Stressed': 0}
            y = np.array([label_map[str(label).strip()] for label in y])
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Initialize and train Random Forest
            self.model = RandomForestClassifier(
                n_estimators=50,  # Reduced from 200
                max_depth=10,      # Reduced from 15
                min_samples_split=10,  # Increased from 5
                random_state=42,
                class_weight='balanced'
            )

            
            logger.info("Training Random Forest model...")
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            logger.info(f"Training accuracy: {train_score:.4f}")
            logger.info(f"Test accuracy: {test_score:.4f}")
            
            # Save model and scaler
            joblib.dump(self.model, MODEL_PATH)
            joblib.dump(self.scaler, SCALER_PATH)
            logger.info("Model and scaler saved successfully")
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise

    def predict_realtime(self, eeg_data):
        """Real-time emotion prediction from EEG data"""
        try:
            # Convert and reshape input data
            eeg_data = np.ascontiguousarray(eeg_data, dtype=np.float32)
            
            # Handle different input shapes
            if len(eeg_data.shape) == 1:  # Single sample (2,)
                eeg_data = eeg_data.reshape(1, 2)
            elif len(eeg_data.shape) == 2:
                if eeg_data.shape[1] != 2:  # If channels are not in second dimension
                    eeg_data = eeg_data.reshape(-1, 2)
            
            # Extract features from the entire buffer
            features = self.extract_frequency_features(eeg_data)
            features = features.reshape(1, -1)  # Reshape for prediction
            
            # Scale features
            if self.scaler is not None:
                features = self.scaler.transform(features)
            
            # Get prediction and probability
            pred_class = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            confidence = probabilities[pred_class]
            
            # Convert to emotion label
            emotion = "stressed" if pred_class == 1 else "relaxed"
            
            # Add small random variation to prevent getting stuck
            confidence = min(0.95, max(0.05, confidence * (1 + np.random.normal(0, 0.05))))
            
            # Store prediction
            self.last_prediction = (emotion, confidence)
            logger.info(f"Prediction: {emotion}, Confidence: {confidence:.4f}")
            
            return emotion, confidence
                
        except Exception as e:
            logger.error(f"Error in real-time prediction: {str(e)}")
            logger.error(traceback.format_exc())
            return self.last_prediction if hasattr(self, 'last_prediction') else ("neutral", 0.5)

# For direct testing
if __name__ == "__main__":
    try:
        # Initialize classifier
        classifier = EEGEmotionClassifier()
        logger.info("Classifier initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing classifier: {str(e)}") 