import { useEffect, useState } from 'react'
import axios from 'axios'

import Header      from './components/Header'
import Chart       from './components/Chart'
import Stack       from './components/Stack'
import SwappyPanel from './components/SwappyPanel'
import SignalPanel from './components/SignalPanel'
import Performance from './components/Performance'
import Trades      from './components/Trades'

const API    = 'http://localhost:8000'
const WS_URL = 'ws://localhost:8000/ws/feed'

export default function App() {
  const [lastSignal,    setLastSignal]    = useState(null)
  const [openPosition,  setOpenPosition]  = useState(null)

  // Fetch position on mount + poll
  useEffect(() => {
    const fetch = () => axios.get(`${API}/position`).then(r => setOpenPosition(r.data)).catch(() => {})
    fetch()
    const id = setInterval(fetch, 5000)
    return () => clearInterval(id)
  }, [])

  // WebSocket for real-time updates
  useEffect(() => {
    let ws
    const connect = () => {
      ws = new WebSocket(WS_URL)
      ws.onmessage = e => {
        const msg = JSON.parse(e.data)
        if (msg.type === 'signal')   setLastSignal(msg.payload)
        if (msg.type === 'trade')    setOpenPosition(null)
        if (msg.type === 'position') setOpenPosition(msg.payload)
      }
      ws.onclose = () => setTimeout(connect, 3000)
    }
    connect()
    return () => ws?.close()
  }, [])

  return (
    <div className="min-h-screen bg-[#0d0d1a] flex flex-col">
      <Header />

      <main className="flex-1 p-4 grid gap-4" style={{ gridTemplateColumns: '1fr 280px' }}>
        {/* Left column */}
        <div className="flex flex-col gap-4">
          <Chart />
          <Performance />
          <Trades />
        </div>

        {/* Right column */}
        <div className="flex flex-col gap-4">
          <Stack />
          <SignalPanel lastSignal={lastSignal} openPosition={openPosition} />
          <SwappyPanel lastSignal={lastSignal} />
        </div>
      </main>
    </div>
  )
}
