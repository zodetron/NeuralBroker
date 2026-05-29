import { useEffect, useState } from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import axios from 'axios'

const API = 'http://localhost:8000'

export default function Performance() {
  const [stats, setStats]   = useState(null)
  const [equity, setEquity] = useState([])

  useEffect(() => {
    const fetch = () => {
      axios.get(`${API}/status`).then(r => setStats(r.data.account)).catch(() => {})
      axios.get(`${API}/trades?n=200`).then(r => {
        const curve = r.data.map((t, i) => ({
          i, bal: t.balance_after ?? 10000
        })).reverse()
        setEquity(curve)
      }).catch(() => {})
    }
    fetch()
    const id = setInterval(fetch, 10000)
    return () => clearInterval(id)
  }, [])

  const wr = stats?.win_rate ? (stats.win_rate * 100).toFixed(1) : '—'

  return (
    <div className="bg-[#111122] rounded-lg border border-gray-800 p-4">
      <div className="text-xs text-gray-400 font-semibold tracking-wider mb-3">PERFORMANCE</div>

      <div className="grid grid-cols-3 gap-2 mb-4">
        <Stat label="Trades"  value={stats?.total_trades ?? '—'} />
        <Stat label="Win Rate" value={`${wr}%`} />
        <Stat label="Net P&L"  value={stats?.total_net_pnl?.toFixed(2) ?? '—'} color={stats?.total_net_pnl >= 0 ? 'text-green-400' : 'text-red-400'} />
      </div>

      {equity.length > 1 && (
        <ResponsiveContainer width="100%" height={100}>
          <LineChart data={equity}>
            <XAxis dataKey="i" hide />
            <YAxis domain={['auto', 'auto']} hide />
            <Tooltip
              contentStyle={{ background: '#1a1a2e', border: '1px solid #2d2d44', fontSize: 11 }}
              formatter={v => [`$${v.toFixed(2)}`, 'Balance']}
              labelFormatter={() => ''}
            />
            <Line type="monotone" dataKey="bal" stroke="#00d4aa" dot={false} strokeWidth={1.5} />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}

function Stat({ label, value, color = 'text-white' }) {
  return (
    <div className="text-center">
      <div className="text-[10px] text-gray-600 mb-0.5">{label}</div>
      <div className={`text-sm font-semibold ${color}`}>{value}</div>
    </div>
  )
}
