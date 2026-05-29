import { useEffect, useState } from 'react'
import axios from 'axios'

const API = 'http://localhost:8000'

export default function Header() {
  const [status, setStatus] = useState(null)

  useEffect(() => {
    const fetch = () => axios.get(`${API}/status`).then(r => setStatus(r.data)).catch(() => {})
    fetch()
    const id = setInterval(fetch, 5000)
    return () => clearInterval(id)
  }, [])

  const bal  = status?.account?.current_balance?.toLocaleString('en-US', { style: 'currency', currency: 'USD' }) ?? '—'
  const pnl  = status?.account?.total_net_pnl ?? 0
  const live = status?.feed_running

  return (
    <header className="flex items-center justify-between px-6 py-3 border-b border-gray-800 bg-[#0d0d1a]">
      <div className="flex items-center gap-3">
        <span className={`w-2 h-2 rounded-full ${live ? 'bg-green-400 animate-pulse' : 'bg-red-500'}`} />
        <span className="text-sm font-bold tracking-widest text-gray-300">MTF LIVE  ·  BTC/USDT  ·  5M</span>
      </div>
      <div className="flex gap-8 text-xs text-gray-400">
        <div>
          <span className="text-gray-600 mr-1">Balance</span>
          <span className="text-white font-semibold">{bal}</span>
        </div>
        <div>
          <span className="text-gray-600 mr-1">Net P&L</span>
          <span className={`font-semibold ${pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {pnl >= 0 ? '+' : ''}{pnl?.toFixed(2) ?? '—'}
          </span>
        </div>
        <div>
          <span className="text-gray-600 mr-1">Candles</span>
          <span className="text-white">{status?.candle_count?.toLocaleString() ?? '—'}</span>
        </div>
      </div>
    </header>
  )
}
