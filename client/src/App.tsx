import { useState } from 'react'
import Header from './components/Header'
import CreatePrediction from './components/CreatePrediction'
import PredictionsList from './components/PredictionsList'
import ChatInterface from './components/ChatInterface'

function App() {
  const [selectedPredictionId, setSelectedPredictionId] = useState<number | null>(null)

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto p-6">
        <Header />
        
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-[calc(100vh-200px)]">
          <div className="lg:col-span-5 space-y-6">
            <CreatePrediction />
            <PredictionsList 
              selectedPredictionId={selectedPredictionId}
              onSelectPrediction={setSelectedPredictionId}
            />
          </div>

          <div className="lg:col-span-7">
            <ChatInterface selectedPredictionId={selectedPredictionId} />
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
