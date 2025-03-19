import React, { useState, useEffect, useCallback } from 'react';
import {
    Box,
    VStack,
    Input,
    Button,
    Text,
    useToast,
    Flex,
    Avatar,
    HStack,
    IconButton,
    Tooltip
} from '@chakra-ui/react';
import { FaMicrophone, FaStop } from 'react-icons/fa';
import { api } from '../services/api';

// Avatar paths for the profile pictures - images should be in the public/images folder
const BOT_AVATAR = "/images/NeuroSri_logo.png";  // Chatbot's profile picture
const USER_AVATAR = "/images/User_logo.png";     // User's profile picture

// Speech synthesis configuration
const VOICE_LANG = 'en-US';
const VOICE_RATE = 0.9;  // Slightly slower for better clarity
const VOICE_PITCH = 1.1;  // Slightly higher pitch for female voice

// Keep track of speech synthesis state
let speechQueue = [];
let isSpeaking = false;

// Function to process speech queue
const processSpeechQueue = () => {
    if (speechQueue.length === 0 || isSpeaking) return;
    
    isSpeaking = true;
    const utterance = speechQueue[0];
    speechSynthesis.speak(utterance);
};

// Function to get the preferred voice
const getPreferredVoice = () => {
    const voices = window.speechSynthesis.getVoices();
    // Try to find a female English voice in this order:
    // 1. Microsoft Zira (Windows)
    // 2. Google US English Female
    // 3. Samantha (macOS)
    // 4. Any female English voice
    // 5. Any English voice
    // 6. Default voice
    
    const preferredVoices = [
        'Microsoft Zira Desktop',
        'Google US English Female',
        'Samantha'
    ];
    
    // First try to find one of our preferred voices
    for (const preferredVoice of preferredVoices) {
        const voice = voices.find(v => v.name === preferredVoice);
        if (voice) return voice;
    }
    
    // Then try to find any female English voice
    const femaleVoice = voices.find(v => 
        v.lang.includes('en') && v.name.toLowerCase().includes('female')
    );
    if (femaleVoice) return femaleVoice;
    
    // Then try any English voice
    const englishVoice = voices.find(v => v.lang.includes('en'));
    if (englishVoice) return englishVoice;
    
    // Fallback to the first available voice
    return voices[0];
};

// Function to handle speech synthesis
const speakText = (text) => {
    if (!('speechSynthesis' in window)) return;

    // Cancel any ongoing speech and clear queue
    speechSynthesis.cancel();
    speechQueue = [];
    isSpeaking = false;

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = VOICE_LANG;
    utterance.rate = VOICE_RATE;
    utterance.pitch = VOICE_PITCH;
    
    // Set up event handlers
    utterance.onend = () => {
        isSpeaking = false;
        speechQueue.shift(); // Remove the completed utterance
        processSpeechQueue(); // Process next in queue if any
    };

    utterance.onerror = (event) => {
        console.error('Speech synthesis error:', event);
        isSpeaking = false;
        speechQueue.shift();
        processSpeechQueue();
    };

    // Handle Chrome bug where speech stops after ~15 seconds
    utterance.onpause = () => {
        if (speechSynthesis.speaking) {
            speechSynthesis.resume();
        }
    };

    // Set the voice
    const voice = getPreferredVoice();
    if (voice) {
        utterance.voice = voice;
    }

    // Add to queue and process
    speechQueue.push(utterance);
    processSpeechQueue();
};

// Keep speech synthesis active
setInterval(() => {
    if (speechSynthesis.speaking && speechSynthesis.paused) {
        speechSynthesis.resume();
    }
}, 100);

function Message({ text, sender }) {
    // Add speech synthesis to each message
    const speakMessage = useCallback(() => {
        if (sender === 'bot') {
            speakText(text);
        }
    }, [text, sender]);

    return (
        <Flex
            w="100%"
            justify={sender === 'user' ? 'flex-end' : 'flex-start'}
            mb={4}
        >
            <HStack
                spacing={2}
                maxW="80%"
                alignItems="flex-start"
                flexDirection={sender === 'user' ? 'row-reverse' : 'row'}
            >
                <Avatar 
                    size="md"
                    src={sender === 'user' ? USER_AVATAR : BOT_AVATAR}
                    name={sender === 'user' ? 'User' : 'NeuroSri'}
                    bg={sender === 'user' ? 'blue.500' : 'green.500'}
                    showBorder={true}
                    borderColor={sender === 'user' ? 'blue.200' : 'green.200'}
                    borderWidth="2px"
                    ignoreFallback={true}
                />
                <Box
                    onClick={sender === 'bot' ? speakMessage : undefined}
                    cursor={sender === 'bot' ? 'pointer' : 'default'}
                    bg={sender === 'user' ? 'blue.500' : 'gray.100'}
                    color={sender === 'user' ? 'white' : 'black'}
                    px={4}
                    py={3}
                    borderRadius="lg"
                    maxW="100%"
                    position="relative"
                    _hover={sender === 'bot' ? {
                        bg: 'gray.200',
                    } : undefined}
                    title={sender === 'bot' ? 'Click to hear this message' : ''}
                    _before={{
                        content: '""',
                        position: 'absolute',
                        top: '12px',
                        [sender === 'user' ? 'right' : 'left']: '-6px',
                        borderWidth: '6px',
                        borderStyle: 'solid',
                        borderColor: 'transparent',
                        [sender === 'user' ? 'borderLeftColor' : 'borderRightColor']: sender === 'user' ? 'blue.500' : 'gray.100'
                    }}
                >
                    <Text fontSize="md">{text}</Text>
                </Box>
            </HStack>
        </Flex>
    );
}

function ChatWindow({ currentEmotion, emotionData }) {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [hasShownSetup, setHasShownSetup] = useState(false);
    const [hasShownInitial, setHasShownInitial] = useState(false);
    const [isListening, setIsListening] = useState(false);
    const toast = useToast();

    // Initialize speech recognition
    const [recognition, setRecognition] = useState(null);

    useEffect(() => {
        if ('webkitSpeechRecognition' in window) {
            const recognition = new window.webkitSpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = false;
            recognition.lang = VOICE_LANG;

            recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript;
                setInput(transcript);
                // Automatically submit after voice input
                handleSubmit(new Event('submit'), transcript);
            };

            recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                setIsListening(false);
                toast({
                    title: "Voice Input Error",
                    description: "There was an error with voice input. Please try again or use text input.",
                    status: "error",
                    duration: 3000,
                    isClosable: true,
                });
            };

            recognition.onend = () => {
                setIsListening(false);
            };

            setRecognition(recognition);
        }
    }, []);

    // Handle voice input
    const toggleVoiceInput = () => {
        if (!recognition) {
            toast({
                title: "Voice Input Not Available",
                description: "Your browser doesn't support voice input. Please use text input instead.",
                status: "warning",
                duration: 3000,
                isClosable: true,
            });
            return;
        }

        if (isListening) {
            recognition.stop();
        } else {
            recognition.start();
            setIsListening(true);
        }
    };

    // Speak the bot's response
    const speakResponse = useCallback((text) => {
        speakText(text);
    }, []);

    // Handle message submission
    const handleSubmit = async (e, voiceInput = null) => {
        e.preventDefault();
        const messageText = voiceInput || input;
        if (!messageText.trim()) return;

        const userMessage = { text: messageText, sender: 'user' };
        setMessages(prev => [...prev, userMessage]);
        setInput('');

        try {
            const response = await api.sendMessage(messageText);
            
            if (response && response.response) {
                const botMessage = { text: response.response, sender: 'bot' };
                setMessages(prev => [...prev, botMessage]);
                // Automatically speak bot's response
                speakResponse(response.response);
            } else if (response.error) {
                toast({
                    title: "Error",
                    description: response.error,
                    status: "error",
                    duration: 3000,
                    isClosable: true,
                });
            }
        } catch (error) {
            console.error('Error sending message:', error);
            toast({
                title: "Error",
                description: "Failed to send message. Please try again.",
                status: "error",
                duration: 3000,
                isClosable: true,
            });
        }
    };

    // Handle initial setup message
    useEffect(() => {
        if (!hasShownSetup) {
            const setupMessage = { 
                text: "Hello! I'm NeuroSri, your mental health AI counselor. Please wear the EEG headset so I can better understand and support you. I'll be with you in just a moment...", 
                sender: 'bot' 
            };
            setMessages([setupMessage]);
            setHasShownSetup(true);
        }
    }, []);

    // Handle first emotion detection message
    useEffect(() => {
        if (!hasShownInitial && emotionData?.emotion && emotionData.emotion !== 'neutral') {
            const initialMessage = { 
                text: "Great! I can now detect your EEG signals and emotional state. I'm NeuroSri, your AI mental health counselor, and I'm here to support you. Could you tell me your name and a bit about yourself? How are you feeling today?", 
                sender: 'bot' 
            };
            setMessages(prev => [...prev, initialMessage]);
            setHasShownInitial(true);
        }
    }, [emotionData?.emotion, hasShownInitial]);

    // Handle subsequent chat messages
    useEffect(() => {
        if (emotionData?.chat_message && hasShownInitial) {
            const botMessage = { text: emotionData.chat_message, sender: 'bot' };
            setMessages(prev => {
                // Check if this message is already in the list to avoid duplicates
                const isDuplicate = prev.some(msg => 
                    msg.text === botMessage.text && msg.sender === 'bot'
                );
                if (isDuplicate) return prev;
                return [...prev, botMessage];
            });
        }
    }, [emotionData?.chat_message, hasShownInitial]);

    return (
        <Box borderWidth={1} borderRadius="lg" p={4} bg="white" height="600px">
            <VStack h="100%" spacing={4}>
                <Box 
                    flex="1" 
                    w="100%" 
                    overflowY="auto" 
                    css={{
                        '&::-webkit-scrollbar': {
                            width: '4px',
                        },
                        '&::-webkit-scrollbar-track': {
                            width: '6px',
                            background: '#f1f1f1',
                        },
                        '&::-webkit-scrollbar-thumb': {
                            background: '#888',
                            borderRadius: '24px',
                        },
                    }}
                    p={2}
                >
                    {messages.map((message, index) => (
                        <Message 
                            key={index}
                            text={message.text}
                            sender={message.sender}
                        />
                    ))}
                </Box>

                <form onSubmit={handleSubmit} style={{ width: '100%' }}>
                    <HStack>
                        <Input
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Type your message..."
                            size="md"
                        />
                        <Tooltip label={isListening ? "Stop voice input" : "Start voice input"}>
                            <IconButton
                                aria-label="Voice input"
                                icon={isListening ? <FaStop /> : <FaMicrophone />}
                                onClick={toggleVoiceInput}
                                colorScheme={isListening ? "red" : "gray"}
                            />
                        </Tooltip>
                        <Button colorScheme="blue" type="submit">
                            Send
                        </Button>
                    </HStack>
                </form>
            </VStack>
        </Box>
    );
}

export default ChatWindow; 