import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  getPredictions,
  createPrediction,
  getPredictionDetails,
  createChat,
  addMessage,
  getChatMessages,
  getChatDetails,
  getImage,
  textToSpeech
} from './requests'
import type { AddMessageRequest } from './types'

// Query Keys
export const queryKeys = {
  predictions: ['predictions'] as const,
  prediction: (id: number) => ['predictions', id] as const,
  chat: (id: number) => ['chat', id] as const,
  chatMessages: (id: number) => ['chat', id, 'messages'] as const,
  image: (filename: string) => ['image', filename] as const,
}

// Prediction Queries
export const usePredictions = () => {
  return useQuery({
    queryKey: queryKeys.predictions,
    queryFn: getPredictions,
  })
}

export const usePredictionDetails = (predictionId: number) => {
  return useQuery({
    queryKey: queryKeys.prediction(predictionId),
    queryFn: () => getPredictionDetails(predictionId),
    enabled: !!predictionId,
  })
}

// Chat Queries
export const useChatDetails = (chatId: number) => {
  return useQuery({
    queryKey: queryKeys.chat(chatId),
    queryFn: () => getChatDetails(chatId),
    enabled: !!chatId,
  })
}

export const useChatMessages = (chatId: number) => {
  return useQuery({
    queryKey: queryKeys.chatMessages(chatId),
    queryFn: () => getChatMessages(chatId),
    enabled: !!chatId,
  })
}

// Image Query
export const useImage = (filename: string) => {
  return useQuery({
    queryKey: queryKeys.image(filename),
    queryFn: () => getImage(filename),
    enabled: !!filename,
  })
}

// Prediction Mutations
export const useCreatePrediction = () => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: createPrediction,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.predictions })
    },
  })
}

// Chat Mutations
export const useCreateChat = () => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: createChat,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.predictions })
    },
  })
}

export const useAddMessage = () => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ chatId, data }: { chatId: number; data: AddMessageRequest }) =>
      addMessage(chatId, data),
    onSuccess: (data, variables) => {
      queryClient.invalidateQueries({ 
        queryKey: queryKeys.chatMessages(variables.chatId) 
      })
      queryClient.invalidateQueries({ 
        queryKey: queryKeys.chat(variables.chatId) 
      })
      textToSpeech(data.content).then((audioBlob) => {
        const url = URL.createObjectURL(audioBlob)
        new Audio(url).play()
      })
    },
  })
}
