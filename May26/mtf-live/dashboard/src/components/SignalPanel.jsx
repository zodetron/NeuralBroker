export default function SignalPanel({ lastSignal, openPosition }) {
  const hasPosition = openPosition !== null && openPosition !== undefined

  return (
    <div className="bg-[#111122] rounded-lg border border-gray-800 p-4">
      <div className="text-xs text-gray-400 font-semibold tracking-wider mb-3">SIGNAL / POSITION</div>

      {hasPosition ? (
        <PositionView p={openPosition} />
      ) : lastSignal ? (
        <SignalView s={lastSignal} />
      ) : (
        <p className="text-xs text-gray-600">Waiting for signal…</p>
      )}
    </div>
  )
}

function SignalView({ s }) {
  const dir = s.direction === 1 ? 'LONG' : 'SHORT'
  const dirColor = s.direction === 1 ? 'text-green-400' : 'text-red-400'

  return (
    <>
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-gray-300">{s.type}</span>
        <span className={`text-xs font-bold ${dirColor}`}>{dir}</span>
      </div>
      <div className="grid grid-cols-2 gap-y-1 text-xs">
        <span className="text-gray-600">Entry</span><span className="text-right text-white">{s.entry?.toFixed(2)}</span>
        <span className="text-gray-600">Stop</span><span className="text-right text-red-400">{s.stop?.toFixed(2)}</span>
        <span className="text-gray-600">TP</span><span className="text-right text-green-400">{s.tp?.toFixed(2)}</span>
        <span className="text-gray-600">R/R</span><span className="text-right text-yellow-300">{s.rr?.toFixed(2)}</span>
        <span className="text-gray-600">ML</span>
        <span className={`text-right ${s.approved ? 'text-green-400' : 'text-red-400'}`}>
          {(s.raw_score * 100).toFixed(1)}% {s.approved ? '✓' : '✗'}
        </span>
      </div>
    </>
  )
}

function PositionView({ p }) {
  const dir = p.direction === 1 ? 'LONG' : 'SHORT'
  const dirColor = p.direction === 1 ? 'text-green-400' : 'text-red-400'

  return (
    <>
      <div className="flex items-center justify-between mb-2">
        <span className="text-[10px] bg-indigo-900 text-indigo-300 px-2 py-0.5 rounded font-bold">OPEN</span>
        <span className={`text-xs font-bold ${dirColor}`}>{dir}</span>
      </div>
      <div className="grid grid-cols-2 gap-y-1 text-xs">
        <span className="text-gray-600">Entry</span><span className="text-right text-white">{p.entry_price?.toFixed(2)}</span>
        <span className="text-gray-600">Stop</span><span className="text-right text-red-400">{p.stop_price?.toFixed(2)}</span>
        <span className="text-gray-600">TP</span><span className="text-right text-green-400">{p.tp_price?.toFixed(2)}</span>
        <span className="text-gray-600">Bars</span><span className="text-right text-gray-300">{p.bars_held}</span>
        <span className="text-gray-600">Type</span><span className="text-right text-gray-400 text-[10px]">{p.signal_type}</span>
      </div>
    </>
  )
}
