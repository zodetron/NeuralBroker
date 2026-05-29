import { useEffect, useState } from 'react'
import axios from 'axios'

const API = 'http://localhost:8000'

export default function Trades() {
  const [trades, setTrades] = useState([])

  useEffect(() => {
    const fetch = () => axios.get(`${API}/trades?n=30`).then(r => setTrades(r.data)).catch(() => {})
    fetch()
    const id = setInterval(fetch, 10000)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="bg-[#111122] rounded-lg border border-gray-800 p-4">
      <div className="text-xs text-gray-400 font-semibold tracking-wider mb-3">RECENT TRADES</div>

      {trades.length === 0 ? (
        <p className="text-xs text-gray-600">No trades yet</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-gray-600 border-b border-gray-800">
                <th className="text-left pb-1">Type</th>
                <th className="text-left pb-1">Dir</th>
                <th className="text-right pb-1">Entry</th>
                <th className="text-right pb-1">Exit</th>
                <th className="text-right pb-1">P&L</th>
                <th className="text-right pb-1">Exit</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((t, i) => {
                const isWin = t.net_pnl > 0
                return (
                  <tr key={i} className="border-b border-gray-900 hover:bg-[#1a1a2e]">
                    <td className="py-1 text-gray-400 text-[10px]">{t.signal_type}</td>
                    <td className={`py-1 font-bold ${t.direction === 1 ? 'text-green-400' : 'text-red-400'}`}>
                      {t.direction === 1 ? 'L' : 'S'}
                    </td>
                    <td className="py-1 text-right text-gray-300">{t.entry_price?.toFixed(1)}</td>
                    <td className="py-1 text-right text-gray-300">{t.exit_price?.toFixed(1)}</td>
                    <td className={`py-1 text-right font-semibold ${isWin ? 'text-green-400' : 'text-red-400'}`}>
                      {t.net_pnl >= 0 ? '+' : ''}{t.net_pnl?.toFixed(2)}
                    </td>
                    <td className="py-1 text-right text-gray-600 text-[10px]">{t.exit_reason}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
