import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The built dashboard is served by FastAPI itself from backend/static/dashboard,
// so the whole product is one process on one port. `npm run dev` proxies the API
// and the WebSocket to the backend so the dev server behaves like production.
export default defineConfig({
  plugins: [react()],
  // Relative asset URLs. The bundle is served by FastAPI from /dashboard/ in
  // production and may sit behind a path-prefixed proxy in preview; a leading
  // "/assets/…" would 404 in both cases.
  base: './',
  build: {
    outDir: '../backend/static/dashboard',
    emptyOutDir: true,
    sourcemap: false,
    chunkSizeWarningLimit: 1200,
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    allowedHosts: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        ws: true,
      },
      '/telephony': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/health': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/metrics': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
})
