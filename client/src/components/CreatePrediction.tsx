import { useState } from 'react'
import { Card } from './ui/card'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { useCreatePrediction } from '../service/queries'

export default function CreatePrediction() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const createPredictionMutation = useCreatePrediction()

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setSelectedFile(file)
    }
  }

  const handleCreatePrediction = async () => {
    if (!selectedFile) return
    
    try {
      await createPredictionMutation.mutateAsync(selectedFile)
      setSelectedFile(null)
      const fileInput = document.getElementById('file-input') as HTMLInputElement
      if (fileInput) fileInput.value = ''
    } catch (error) {
      console.error('Error creating prediction:', error)
    }
  }

  return (
    <Card className="p-6">
      <h2 className="text-xl font-semibold mb-4">Yeni Tahmin Oluştur</h2>
      <div className="space-y-4">
        <div>
          <label htmlFor="file-input" className="block text-sm font-medium text-gray-700 mb-2">
            Resim Yükleyiniz
          </label>
          <Input
            id="file-input"
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            className="mb-2"
          />
          {selectedFile && (
            <p className="text-sm text-gray-600">Selected: {selectedFile.name}</p>
          )}
        </div>
        <Button 
          onClick={handleCreatePrediction}
          disabled={!selectedFile || createPredictionMutation.isPending}
          className="w-full"
        >
          {createPredictionMutation.isPending ? 'Analiz Ediliyor...' : 'Tahmin Oluştur'}
        </Button>
      </div>
    </Card>
  )
}