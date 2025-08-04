import { useEffect, useRef } from 'react'
import { ScrollArea } from './ui/scroll-area'

interface Message {
  id: number
  content: string
  role: "user" | "assistant"
  created_at: string
}

interface ChatMessagesProps {
  messages: Message[]
}

export default function ChatMessages({ messages }: ChatMessagesProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  return (
    <ScrollArea className="h-[400px] lg:h-[500px] p-4">
      <div className="space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.role == "user" ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[85%] sm:max-w-[80%] p-3 rounded-lg ${
                message.role == "user"
                  ? 'bg-blue-600 text-white ml-4'
                  : 'bg-gray-100 text-gray-900 mr-4'
              }`}
            >
              <p className="whitespace-pre-wrap">{message.content}</p>
              <div className={`text-xs mt-2 ${
                message.role == "user" ? 'text-blue-100' : 'text-gray-500'
              }`}>
                {formatDate(message.created_at)}
              </div>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
    </ScrollArea>
  )
}