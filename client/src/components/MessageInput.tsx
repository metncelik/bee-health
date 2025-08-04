import { useState, useRef } from 'react'
import { Button } from './ui/button'
import { Textarea } from './ui/textarea'
import { useAddMessage } from '../service/queries'
import { speechToText } from '../service/requests'

interface MessageInputProps {
  chatId: number
}

export default function MessageInput({ chatId }: MessageInputProps) {
  const [newMessage, setNewMessage] = useState('')
  const [isRecording, setIsRecording] = useState(false)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])

  const addMessageMutation = useAddMessage()

  const handleSendMessage = async () => {
    if (!newMessage.trim()) return

    try {
      const messageContent = newMessage
      await addMessageMutation.mutateAsync({
        chatId,
        data: { content: messageContent }
      })
      setNewMessage('')
    } catch (error) {
      console.error('Error sending message:', error)
    }
  }

  const handleRecordClick = async () => {
    if (isRecording) {
      // Stop recording
      mediaRecorderRef.current?.stop()
      setIsRecording(false)
      return
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)
      mediaRecorderRef.current = recorder
      chunksRef.current = []

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data)
        }
      }

      recorder.onstop = async () => {
        const audioBlob = new Blob(chunksRef.current, { type: 'audio/webm' })
        try {
          const { text } = await speechToText(audioBlob)
          setNewMessage((prev) => prev + text)
        } catch (err) {
          console.error('Speech to text error:', err)
        }
      }

      recorder.start()
      setIsRecording(true)
    } catch (err) {
      console.error('Microphone access error:', err)
    }
  }

  return (
    <div className="p-4 border-t">
      <div className="flex flex-col sm:flex-row gap-2">
        <Textarea
          value={newMessage}
          onChange={(e) => setNewMessage(e.target.value)}
          placeholder="Bir soru sorunuz..."
          className="flex-1 min-h-[60px] resize-none sm:min-h-[80px]"
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              handleSendMessage()
            }
          }}
        />
        <div className="flex gap-2 sm:self-end w-full sm:w-auto">
          <Button
            variant="secondary"
            onClick={handleRecordClick}
            className="w-full sm:w-auto"
          >
            {isRecording ? 'Durdur' : '🎤'}
          </Button>
          <Button
            onClick={handleSendMessage}
            disabled={!newMessage.trim() || addMessageMutation.isPending}
            className="w-full sm:w-auto"
          >
            {addMessageMutation.isPending ? 'Gönderiliyor...' : 'Gönder'}
          </Button>
        </div>
      </div>
    </div>
  )
}
