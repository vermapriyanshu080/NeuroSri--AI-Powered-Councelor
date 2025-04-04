import axios from 'axios';

// Get current hostname and protocol from window location
const currentHostname = window.location.hostname;
const currentProtocol = window.location.protocol;

// Use window.location to get the current hostname for local development
const API_URL = `${currentProtocol}//${currentHostname}:5000/api`;

// Create axios instance with proper baseURL
const axiosInstance = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json',
    },
    timeout: 10000 // 10 second timeout
});

export const api = {
    getEmotion: async () => {
        try {
            const response = await axiosInstance.get('/emotion');
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
            return {
                emotion: 'neutral',
                confidence: 0,
                eeg_data: [0, 0],
                error: 'Could not connect to EEG service'
            };
        }
    },

    sendMessage: async (message, userInfo = null) => {
        try {
            const response = await axiosInstance.post('/chat', { 
                message,
                userInfo
            });
            return response.data;
        } catch (error) {
            console.error('API Error:', error);
            return { error: 'Failed to communicate with the server.' };
        }
    },

    submitUserInfo: async (userInfo) => {
        try {
            const response = await axiosInstance.post('/user/info', userInfo);
            return response.data;
        } catch (error) {
            console.error('API Error:', error);
            return { error: 'Failed to save user information.', success: false };
        }
    },

    getEEGData: async () => {
        try {
            const response = await axiosInstance.get('/eeg-data');
            return response.data;
        } catch (error) {
            console.error('API Error:', error);
            return { error: 'Failed to fetch EEG data.' };
        }
    }
}; 