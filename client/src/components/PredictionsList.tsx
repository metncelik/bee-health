import { Card } from './ui/card'
import { Badge } from './ui/badge'
import { ScrollArea } from './ui/scroll-area'
import { usePredictions } from '../service/queries'

interface PredictionsListProps {
    selectedPredictionId: number | null
    onSelectPrediction: (id: number) => void
}

export default function PredictionsList({ selectedPredictionId, onSelectPrediction }: PredictionsListProps) {
    const { data: predictions = [], isLoading: predictionsLoading } = usePredictions()

    const formatDate = (dateString: string) => {
        return new Date(dateString).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        })
    }

    const getHealthStatusColor = (className: string) => {
        const lowerName = className.toLowerCase()
        if (lowerName.includes('healthy') || lowerName.includes('good')) return 'bg-green-100 text-green-800'
        if (lowerName.includes('warning') || lowerName.includes('moderate')) return 'bg-yellow-100 text-yellow-800'
        if (lowerName.includes('disease') || lowerName.includes('poor')) return 'bg-red-100 text-red-800'
        return 'bg-gray-100 text-gray-800'
    }

    return (
        <Card className="p-6 flex-1">
            <h2 className="text-xl font-semibold mb-4">Tahminler</h2>
            <ScrollArea className="h-[300px] lg:h-[400px]">
                {predictionsLoading ? (
                    <div className="text-center py-8 text-gray-500">Yükleniyor...</div>
                ) : predictions.length === 0 ? (
                    <div className="text-center py-8 text-gray-500">Henüz tahmin yok. Resim yükleyiniz!</div>
                ) : (
                    <div className="space-y-3">
                        {predictions.map((prediction) => (
                            <Card
                                key={prediction.id}
                                className={`p-4 m-2 cursor-pointer transition-colors hover:bg-gray-50 ${selectedPredictionId === prediction.id ? 'ring-2 ring-blue-500 bg-blue-50' : ''
                                    }`}
                                onClick={() => onSelectPrediction(prediction.id)}
                            >
                                <div className="flex gap-4 items-start mb-2">

                                    {prediction.image && (
                                        <div className="mb-3 flex justify-center">
                                            <img
                                                src={`${import.meta.env.VITE_API_URL}/${prediction.image.url}`}
                                                alt={`Bee health prediction ${prediction.id}`}
                                                className="w-20 h-20 object-cover rounded-md border"
                                            />
                                        </div>
                                    )}
                                    <div className="flex flex-col gap-2">
                                        <div className="flex items-center gap-2">
                                            <span className="font-medium">Tahmin #{prediction.id}</span>
                                            <Badge className={getHealthStatusColor(prediction.class.name)}>
                                                {prediction.class.name}
                                            </Badge>
                                        </div>
                                        <span className="text-xs text-gray-500">
                                            {formatDate(prediction.created_at)}
                                        </span>
                                    </div>
                                </div>
                            </Card>
                        ))}
                    </div>
                )}
            </ScrollArea>
        </Card>
    )
}