import axios from 'axios'
import type { 
  PredictionListItem,
  PredictionWithDetails, 
  PredictionResult, 
  Chat, 
  ChatMessage, 
  ChatDetails,
  CreateChatRequest,
  AddMessageRequest 
} from './types'

const API_URL = import.meta.env.VITE_API_URL

// Prediction requests
export const getPredictions = async (): Promise<PredictionListItem[]> => {
  const response = await axios.get(`${API_URL}/predictions`)
  return response.data
}

export const createPrediction = async (image: File): Promise<PredictionResult> => {
  const formData = new FormData()
  formData.append('image', image)
  
  const response = await axios.post(`${API_URL}/predictions`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}

export const getPredictionDetails = async (predictionId: number): Promise<PredictionWithDetails> => {
  const response = await axios.get(`${API_URL}/predictions/${predictionId}`)
  return response.data
}

// Chat requests
export const createChat = async (data: CreateChatRequest): Promise<Chat> => {
  const response = await axios.post(`${API_URL}/chat`, data)
  return response.data
}

export const addMessage = async (chatId: number, data: AddMessageRequest): Promise<ChatMessage> => {
  const response = await axios.post(`${API_URL}/chat/${chatId}/messages`, data)
  return response.data
}

export const getChatMessages = async (chatId: number): Promise<ChatMessage[]> => {
  const response = await axios.get(`${API_URL}/chat/${chatId}/messages`)
  return response.data
}

export const getChatDetails = async (chatId: number): Promise<ChatDetails> => {
  const response = await axios.get(`${API_URL}/chat/${chatId}`)
  return response.data
}

// Image requests
export const getImage = async (filename: string): Promise<Blob> => {
  const response = await axios.get(`${API_URL}/images/${filename}`, {
    responseType: 'blob',
  })
  return response.data
}

// Speech requests
export const speechToText = async (audio: Blob): Promise<{ text: string }> => {
  const formData = new FormData()
  formData.append('audio_file', audio, 'recording.webm')

  const response = await axios.post(`${API_URL}/speech/speech-to-text`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  return response.data
}

export const textToSpeech = async (text: string): Promise<Blob> => {
  const response = await axios.post(
    `${API_URL}/speech/text-to-speech`,
    { text },
    {
      responseType: 'blob',
    }
  )
  return response.data as Blob
}