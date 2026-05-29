import { useEffect, useRef, useState } from 'react'
import { createChart, CandlestickSeries } from 'lightweight-charts'
import axios from 'axios'

const API = 'http://localhost:8000'

const TF_OPTIONS = [
  { label: '5M',  tf: 5    },
  { label: '15M', tf: 15   },
  { label: '1H',  tf: 60   },
  { label: '4H',  tf: 240  },
  { label: '1D',  tf: 1440 },
]

export default function Chart() {
  const containerRef = useRef(null)
  const chartRef     = useRef(null)
  const seriesRef    = useRef(null)
  const [activeTf, setActiveTf] = useState(5)

  useEffect(() => {
    if (!containerRef.current) return

    const chart = createChart(containerRef.current, {
      layout:     { background: { color: '#0d0d1a' }, textColor: '#9ca3af' },
      grid:       { vertLines: { color: '#1a1a2e' }, horzLines: { color: '#1a1a2e' } },
      crosshair:  { mode: 1 },
      timeScale:  { borderColor: '#2d2d44', timeVisible: true, secondsVisible: false },
      rightPriceScale: { borderColor: '#2d2d44' },
      width:  containerRef.current.clientWidth,
      height: 340,
    })

    // v5 API: addSeries(SeriesType, options)
    const series = chart.addSeries(CandlestickSeries, {
      upColor:       '#00d4aa',
      downColor:     '#ff4d6d',
      borderVisible: false,
      wickUpColor:   '#00d4aa',
      wickDownColor: '#ff4d6d',
    })

    chartRef.current  = chart
    seriesRef.current = series

    const ro = new ResizeObserver(entries => {
      chart.applyOptions({ width: entries[0].contentRect.width })
    })
    ro.observe(containerRef.current)

    return () => { ro.disconnect(); chart.remove() }
  }, [])

  useEffect(() => {
    if (!seriesRef.current) return

    axios.get(`${API}/candles/${activeTf}?n=300`)
      .then(r => {
        const data = r.data.map(b => ({
          time:  Math.floor(new Date(b.timestamp).getTime() / 1000),
          open:  b.open,
          high:  b.high,
          low:   b.low,
          close: b.close,
        }))
        seriesRef.current.setData(data)
        chartRef.current?.timeScale().fitContent()
      })
      .catch(() => {})
  }, [activeTf])

  return (
    <div className="bg-[#111122] rounded-lg border border-gray-800 overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2 border-b border-gray-800">
        <span className="text-xs text-gray-400 font-semibold tracking-wider">PRICE CHART</span>
        <div className="flex gap-1">
          {TF_OPTIONS.map(o => (
            <button
              key={o.tf}
              onClick={() => setActiveTf(o.tf)}
              className={`px-2 py-0.5 text-xs rounded transition-colors ${
                activeTf === o.tf
                  ? 'bg-indigo-600 text-white'
                  : 'text-gray-500 hover:text-gray-300'
              }`}
            >
              {o.label}
            </button>
          ))}
        </div>
      </div>
      <div ref={containerRef} className="w-full" />
    </div>
  )
}
