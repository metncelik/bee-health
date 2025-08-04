import { Card } from './ui/card'
import { usePredictionDetails, useChatMessages } from '../service/queries'
import ChatHeader from './ChatHeader'
import ChatMessages from './ChatMessages'
import MessageInput from './MessageInput'
import StartChatSection from './StartChatSection'
import NoPredictionSelected from './NoPredictionSelected'

interface ChatInterfaceProps {
  selectedPredictionId: number | null
}

export default function ChatInterface({ selectedPredictionId }: ChatInterfaceProps) {
  const { data: selectedPrediction } = usePredictionDetails(selectedPredictionId || 0)
  const { data: chatMessages = [] } = useChatMessages(selectedPrediction?.chat?.id || 0)

  return (
    <Card className="h-full flex flex-col">
      {selectedPredictionId && selectedPrediction ? (
        <>
          <ChatHeader
            predictionId={selectedPrediction.prediction.id}
            className={selectedPrediction.class.name}
            confidence={selectedPrediction.prediction.confidence}
          />

          {selectedPrediction.chat ? (
            <>
              <ChatMessages messages={chatMessages} />
              <MessageInput chatId={selectedPrediction.chat.id} />
            </>
          ) : (
            <StartChatSection selectedPrediction={selectedPrediction} />
          )}
        </>
      ) : (
        <NoPredictionSelected />
      )}
    </Card>
  )
}