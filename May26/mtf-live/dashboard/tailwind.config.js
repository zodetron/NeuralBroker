/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          green:  "#00d4aa",
          red:    "#ff4d6d",
          yellow: "#ffd166",
          blue:   "#4cc9f0",
          gray:   "#1e1e2e",
        },
      },
    },
  },
  plugins: [],
}
