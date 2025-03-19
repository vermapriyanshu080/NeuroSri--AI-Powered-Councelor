import React, { useState, useEffect } from 'react';
import { ChakraProvider, Box, VStack, Container } from '@chakra-ui/react';
import EEGDisplay from './components/EEGDisplay';
import ChatWindow from './components/ChatWindow';
import EmotionDisplay from './components/EmotionDisplay';
import { api } from './services/api';

function App() {
  const [emotionData, setEmotionData] = useState({
    emotion: null,
    confidence: null,
    chat_message: null,
    is_setup_phase: true,
    setup_complete: false
  });
  const [eegData, setEEGData] = useState([]);
  
  // Poll emotion data
  useEffect(() => {
    const pollEmotion = async () => {
      try {
        const data = await api.getEmotion();
        setEmotionData(data);
        setEEGData(data.eeg_data || []);
      } catch (error) {
        console.error('Error fetching emotion:', error);
      }
    };

    const interval = setInterval(pollEmotion, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <ChakraProvider>
      <Container maxW="container.xl" py={5}>
        <VStack spacing={5}>
          <Box w="100%">
            <EEGDisplay data={eegData} />
          </Box>
          <Box w="100%">
            <EmotionDisplay 
              emotion={emotionData.emotion} 
              confidence={emotionData.confidence} 
            />
          </Box>
          <Box w="100%">
            <ChatWindow 
              currentEmotion={emotionData.emotion} 
              emotionData={emotionData}
            />
          </Box>
        </VStack>
      </Container>
    </ChakraProvider>
  );
}

export default App; 