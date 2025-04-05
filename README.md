# NeuroSri: EEG-Based Mental Health Chatbot 🧠💬

## Overview

NeuroSri is a cutting-edge AI mental health assistant that integrates real-time EEG (electroencephalogram) data with advanced natural language processing to deliver personalized therapeutic interactions. The system monitors brain activity to assess a user's emotional state and tailors therapeutic responses accordingly, creating a more emotionally intelligent AI counseling experience.

![NeuroSri Interface](https://github.com/user-attachments/assets/53559991-9800-4217-baa9-f8c2abea47d5)


## ✨ Key Features

- **Real-time EEG Analysis**: Processes brain activity to identify emotional states with confidence scores
- **Emotion-Aware Responses**: Adapts therapeutic approach based on detected emotions
- **Personalized Experience**: Uses user profile information to customize interactions
- **Interactive UI**: Clean, modern interface showing both conversation and brain activity
- **Conversation History**: Download chat logs for record-keeping or further analysis
- **Female Therapeutic Persona**: Consistent feminine voice with nurturing, compassionate communication style

## 🏗️ System Architecture

NeuroSri follows a multi-layered architecture that integrates hardware data acquisition, signal processing, machine learning, and natural language processing:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │     │                 │
│  EEG Headset    │──→ │  Signal         │──→ │  Emotion        │──→ │  Therapeutic    │
│  Data Stream    │     │  Processing     │     │  Classification  │     │  Chatbot        │
│                 │     │                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
                                                                               │
                                                                               ▼
                                                                        ┌─────────────────┐
                                                                        │                 │
                                                                        │  Web Interface  │
                                                                        │                 │
                                                                        └─────────────────┘
```

### Technical Components

The system consists of five primary components:

1. **Data Acquisition (CHORDS Stream)**
   - Interfaces with EEG hardware
   - Streams raw brainwave data in real-time
   - Provides data buffering and preprocessing

2. **Signal Processing**
   - Filters and cleans raw EEG signals
   - Performs feature extraction (FFT, wavelet transforms)
   - Prepares data for emotion classification

3. **Emotion Classification**
   - Uses CNN-LSTM hybrid neural network
   - Classifies emotional states (relaxed, stressed, etc.)
   - Provides confidence scores for detected emotions

4. **Therapeutic Chatbot**
   - Incorporates detected emotions into responses
   - Maintains conversational context and user profile
   - Adapts linguistic style based on emotional state

5. **Web Interface**
   - React-based frontend with real-time updating
   - Displays conversation and EEG/emotion data
   - Provides user profile management and session controls

## 📂 File Structure

```
NeuroSri/
├── server.py                 # Main Flask server
├── src/
│   ├── app.py                # Application entry point
│   ├── chatbot/              # Therapeutic bot components
│   │   ├── chatbot_service.py # Core chatbot functionality
│   │   └── therapeutic_bot.py # Therapeutic response generation
│   ├── chords/               # EEG data streaming
│   ├── components/           # React UI components
│   │   └── ChatWindow.tsx    # Main chat interface
│   ├── config/               # Configuration settings
│   ├── eeg/                  # EEG signal processing
│   │   ├── data_acquisition.py # Raw data handling
│   │   └── signal_processing.py # Signal filtering and processing
│   └── ml/                   # Machine learning models
│       ├── deep_emotion_model.py # Neural network architecture
│       └── emotion_classifier.py # Emotion classification logic
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Node.js 14+
- EEG headset compatible with the CHORDS framework
- Required Python packages (see `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/neurosri.git
   cd neurosri
   ```

2. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install frontend dependencies:
   ```bash
   cd frontend
   npm install
   ```

4. Configure your EEG device in `src/config/settings.py`

### Running the Application

1. Start the backend server:
   ```bash
   python server.py
   ```

2. Start the frontend development server:
   ```bash
   cd frontend
   npm start
   ```

3. Navigate to `http://localhost:3000` in your browser

## 📊 How It Works

### User Session Flow

1. **Initial Setup**:
   - User creates profile with personal information
   - User connects EEG headset
   - System calibrates and begins monitoring brain activity

2. **Conversation**:
   - User engages in conversation with NeuroSri
   - EEG data is continuously processed (every 5 seconds)
   - Emotional state is classified with confidence scores

3. **Therapeutic Response**:
   - Chatbot generates responses based on:
     - User's message content
     - Detected emotional state
     - Personal profile information
     - Conversation history

4. **Feedback Loop**:
   - System tracks emotional transitions
   - Provides context-aware therapeutic suggestions
   - Adapts communication style to match emotional needs

### API Endpoints

- `/api/chat` - Main endpoint for conversation interaction
- `/api/chat/download` - Download conversation history
- `/api/user/info` - Submit/update user profile information
- `/api/emotion` - Get current emotion classification
- `/ws/emotions` - WebSocket for real-time emotion updates

## 🔧 Advanced Configuration

### Emotion Detection Sensitivity

Adjust the emotion detection threshold in `src/config/settings.py`:

```python
# Higher values require more confident predictions
EMOTION_CONFIDENCE_THRESHOLD = 0.65

# Time interval between emotion readings (seconds)
PREDICTION_INTERVAL = 5
```

### Therapeutic Style

Modify the chatbot's therapeutic approach in `src/chatbot/therapeutic_bot.py`:

```python
# Adjust emotion-specific guidance for different therapeutic approaches
emotion_specific_guidance = {
    'stressed': "Focus on calming techniques and stress relief strategies...",
    'sad': "Offer emotional support and validation with a warm approach...",
    # Add or customize approaches for different emotions
}
```

## 📱 Mobile EEG Integration

NeuroSri supports several consumer-grade EEG headsets:

- Muse 2
- Emotiv EPOC X
- NeuroSky MindWave
- OpenBCI Ganglion

Configure your device in `src/config/devices.py`.

## 🤝 Contributing

Contributions to NeuroSri are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- NeuroEngineers team for developing the core technology
- OpenAI for language model assistance
- CHORDS framework developers for EEG signal processing
- All mental health professionals who provided guidance on therapeutic approaches

---

*NeuroSri is intended as a supplemental mental wellness tool and is not a replacement for professional mental health services.* 
