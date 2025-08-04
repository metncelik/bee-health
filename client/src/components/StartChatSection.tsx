import { Button } from './ui/button'
import { Badge } from './ui/badge'
import { useCreateChat } from '../service/queries'

interface PredictionDetails {
  prediction: {
    id: number
    confidence: number
  }
  class: {
    name: string
    description: string
  }
}

interface StartChatSectionProps {
  selectedPrediction: PredictionDetails
}

export default function StartChatSection({ selectedPrediction }: StartChatSectionProps) {
  const createChatMutation = useCreateChat()

  const handleStartChat = async () => {
    try {
      await createChatMutation.mutateAsync({ prediction_id: selectedPrediction.prediction.id })
    } catch (error) {
      console.error('Error creating chat:', error)
    }
  }

  const getHealthStatusColor = (className: string) => {
    const lowerName = className.toLowerCase()
    if (lowerName.includes('healthy') || lowerName.includes('good')) return 'bg-green-100 text-green-800'
    if (lowerName.includes('warning') || lowerName.includes('moderate')) return 'bg-yellow-100 text-yellow-800'
    if (lowerName.includes('disease') || lowerName.includes('poor')) return 'bg-red-100 text-red-800'
    return 'bg-gray-100 text-gray-800'
  }

  return (
    <div className="flex-1 flex items-center justify-center p-8">
      <div className="text-center">
        <div className="mb-6">
          <h3 className="text-lg font-semibold mb-2">Yapay Zeka Asistanınıza Sorun</h3>
          <p className="text-gray-600 mb-4">
           Arı kovanınız hakkında sorularınızı sorabilir veya öneriler alabilirsiniz.
          </p>
          <div className="bg-gray-50 p-4 rounded-lg mb-4">
            <h4 className="font-medium mb-2">Analiz Sonuçları:</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>Sağlık Durumu:</span>
                <Badge className={getHealthStatusColor(selectedPrediction.class.name)}>
                  {selectedPrediction.class.name}
                </Badge>
              </div>
              <div className="flex justify-between">
                <span>Güven:</span>
                <span>{(selectedPrediction.prediction.confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="text-left mt-3">
                <span className="font-medium">Açıklama:</span>
                <p className="text-gray-600 mt-1">{selectedPrediction.class.description}</p>
              </div>
            </div>
          </div>
        </div>
        <Button 
          onClick={handleStartChat}
          disabled={createChatMutation.isPending}
          size="lg"
          >
            {createChatMutation.isPending ? 'Sohbet Başlatılıyor...' : 'Sohbet Başlat'}
        </Button>
      </div>
    </div>
  )
}