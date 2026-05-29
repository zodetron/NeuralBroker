import { useEffect, useState } from 'react'
import axios from 'axios'

const API = 'http://localhost:8000'

const BADGE_COLORS = {
  TRENDING_UP:    'bg-green-900 text-green-300',
  TRENDING_DOWN:  'bg-red-900 text-red-300',
  BREAKOUT_UP:    'bg-emerald-900 text-emerald-300',
  BREAKOUT_DOWN:  'bg-rose-900 text-rose-300',
  RANGING:        'bg-gray-700 text-gray-300',
  STRONG_BULL:    'bg-green-900 text-green-300',
  PULLBACK_BULL:  'bg-teal-900 text-teal-300',
  STRONG_BEAR:    'bg-red-900 text-red-300',
  PULLBACK_BEAR:  'bg-orange-900 text-orange-300',
  EXHAUSTED:      'bg-yellow-900 text-yellow-300',
  NEUTRAL:        'bg-gray-700 text-gray-300',
  DEMAND:         'bg-green-900 text-green-300',
  SUPPLY:         'bg-red-900 text-red-300',
  MIDZONE:        'bg-gray-700 text-gray-300',
}

function Badge({ label }) {
  const cls = BADGE_COLORS[label] ?? 'bg-gray-700 text-gray-400'
  return <span className={`px-2 py-0.5 rounded text-[10px] font-bold tracking-wider ${cls}`}>{label}</span>
}

function Row({ tf, label, value, extra }) {
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-gray-800 last:border-0">
      <div className="flex items-center gap-2">
        <span className="text-[10px] text-gray-600 w-8">{tf}</span>
        <span className="text-xs text-gray-400">{label}</span>
      </div>
      <div className="flex items-center gap-2">
        {extra && <span className="text-[10px] text-gray-600">{extra}</span>}
        {typeof value === 'string' ? <Badge label={value} /> : <span className="text-xs text-white">{value}</span>}
      </div>
    </div>
  )
}

export default function Stack() {
  const [stack, setStack] = useState(null)

  useEffect(() => {
    const fetch = () => axios.get(`${API}/stack`).then(r => setStack(r.data)).catch(() => {})
    fetch()
    const id = setInterval(fetch, 5000)
    return () => clearInterval(id)
  }, [])

  const d1  = stack?.['1d']  ?? {}
  const d4h = stack?.['4h']  ?? {}
  const d1h = stack?.['1h']  ?? {}
  const d15 = stack?.['15m'] ?? {}
  const d5  = stack?.['5m']  ?? {}

  return (
    <div className="bg-[#111122] rounded-lg border border-gray-800 p-4">
      <div className="text-xs text-gray-400 font-semibold tracking-wider mb-3">TF STACK</div>
      <Row tf="1D"  label="Bias"      value={d1.bias === 1 ? 'BULLISH' : d1.bias === -1 ? 'BEARISH' : 'NEUTRAL'} extra={`SMA200: ${d1.sma200?.toFixed(0) ?? '—'}`} />
      <Row tf="4H"  label="Zone"      value={d4h.zone ?? '—'} extra={`S: ${d4h.nearest_support?.toFixed(0) ?? '—'}`} />
      <Row tf="1H"  label="Structure" value={d1h.structure ?? '—'} extra={`ADX: ${d1h.adx?.toFixed(1) ?? '—'}`} />
      <Row tf="15M" label="Momentum"  value={d15.momentum ?? '—'} extra={`RSI: ${d15.rsi?.toFixed(1) ?? '—'}`} />
      <Row tf="5M"  label="Trigger"   value={d5.trigger ?? 'NONE'} extra={d5.direction === 1 ? 'LONG' : d5.direction === -1 ? 'SHORT' : ''} />
    </div>
  )
}
