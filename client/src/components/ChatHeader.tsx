import { Badge } from './ui/badge'

interface ChatHeaderProps {
  predictionId: number
  className: string
  confidence: number
}

export default function ChatHeader({ predictionId, className, confidence }: ChatHeaderProps) {
  const getHealthStatusColor = (className: string) => {
    const lowerName = className.toLowerCase()
    if (lowerName.includes('healthy') || lowerName.includes('good')) return 'bg-green-100 text-green-800'
    if (lowerName.includes('warning') || lowerName.includes('moderate')) return 'bg-yellow-100 text-yellow-800'
    if (lowerName.includes('disease') || lowerName.includes('poor')) return 'bg-red-100 text-red-800'
    return 'bg-gray-100 text-gray-800'
  }

  return (
    <div className="p-4 border-b">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="font-semibold">
            Tahmin #{predictionId} - {className}
          </h3>
          <p className="text-sm text-gray-600">
            Arıcılıkta sağlık konularında size yardımcı olur.
          </p>
        </div>
        <Badge className={getHealthStatusColor(className)}>
          {(confidence * 100).toFixed(1)}% güven
        </Badge>
      </div>
    </div>
  )
}