import { useEffect, useState } from 'react'

export default function SwappyPanel({ lastSignal }) {
  if (!lastSignal || !lastSignal.type?.startsWith('SWAPPY')) {
    return (
      <div className="bg-[#111122] rounded-lg border border-gray-800 p-4">
        <div className="text-xs text-gray-400 font-semibold tracking-wider mb-2">SWAPPY ICT</div>
        <p className="text-xs text-gray-600">No active Swappy setup</p>
      </div>
    )
  }

  const s   = lastSignal
  const dir = s.direction === 1 ? 'LONG' : 'SHORT'
  const dirColor = s.direction === 1 ? 'text-green-400' : 'text-red-400'
  const confBadge = s.confluence
    ? <span className="px-1.5 py-0.5 rounded text-[10px] bg-yellow-900 text-yellow-300 font-bold">HIGH CONF</span>
    : null

  return (
    <div className="bg-[#111122] rounded-lg border border-gray-800 p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs text-gray-400 font-semibold tracking-wider">SWAPPY ICT</span>
        <div className="flex gap-2 items-center">
          {confBadge}
          <span className={`text-xs font-bold ${dirColor}`}>{dir}</span>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-y-1.5 text-xs">
        <span className="text-gray-600">Entry</span>
        <span className="text-white text-right">{s.entry?.toFixed(2)}</span>
        <span className="text-gray-600">Stop</span>
        <span className="text-red-400 text-right">{s.stop?.toFixed(2)}</span>
        <span className="text-gray-600">TP</span>
        <span className="text-green-400 text-right">{s.tp?.toFixed(2)}</span>
        <span className="text-gray-600">R/R</span>
        <span className="text-yellow-300 text-right">{s.rr?.toFixed(2)}</span>
        <span className="text-gray-600">ML Score</span>
        <span className={`text-right ${s.raw_score >= 0.5 ? 'text-green-400' : 'text-gray-500'}`}>
          {(s.raw_score * 100).toFixed(1)}%
        </span>
        <span className="text-gray-600">Setup at</span>
        <span className="text-gray-400 text-right text-[10px]">
          {s.setup_ts ? new Date(s.setup_ts).toUTCString().slice(5, 22) : '—'}
        </span>
      </div>
    </div>
  )
}
