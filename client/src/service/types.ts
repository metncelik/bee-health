// Prediction Types
export type PredictionListItem = {
    id: number
    image_id: number
    confidence: number
    created_at: string
    class: {
        id: number
        name: string
        description: string
    }
    image: {
        id: number
        url: string
    }
    chat: {
        id: number
        prediction_id: number
        created_at: string
    } | null
}

export type PredictionWithDetails = {
    prediction: {
        id: number
        image_id: number
        confidence: number
        created_at: string
    }
    class: {
        id: number
        name: string
        description: string
        created_at: string
    }
    image: {
        id: number
        url: string
        created_at: string
    }
    chat: {
        id: number
        prediction_id: number
        created_at: string
    } | null
}

export type PredictionResult = {
    prediction_id: number
    image_id: number
    predicted_class: string
    confidence: number
    all_probabilities: Record<string, number>
}

// Chat Types
export type Chat = {
    id: number
    prediction_id: number
    created_at: string
}

export type ChatMessage = {
    id: number
    chat_id: number
    content: string
    role: "user" | "assistant"
    created_at: string
}

export type ChatDetails = {
    chat: Chat
    prediction: PredictionWithDetails
    messages: ChatMessage[]
}

// Request Types
export type CreateChatRequest = {
    prediction_id: number
}

export type AddMessageRequest = {
    content: string
}