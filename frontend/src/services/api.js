import axios from 'axios';

// Use window.location to get the current hostname
const API_URL = `http://${window.location.hostname}:5000/api`;

export const api = {
    getEmotion: async () => {
        try {
            const response = await axios.get(`${API_URL}/emotion`);
            return {
                emotion: response.data.emotion,
                confidence: response.data.confidence,
                chat_message: response.data.chat_message,
                is_setup_phase: response.data.is_setup_phase,
                setup_complete: response.data.setup_complete,
                eeg_data: response.data.eeg_data || [0, 0]
            };
        } catch (error) {
            console.error('Error fetching emotion:', error);
            throw error;
        }
    },

    sendMessage: async (message) => {
        try {
            const response = await axios.post(`${API_URL}/chat`, { message });
            return response.data;
        } catch (error) {
            console.error('Error sending message:', error);
            throw error;
        }
    }
}; 