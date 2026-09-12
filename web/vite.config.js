import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  base: './',
  plugins: [tailwindcss(), svelte()],
  server: {
    proxy: { '/api': 'http://localhost:7860' }
  },
  build: {
    outDir: 'dist',
    chunkSizeWarningLimit: 1024
  }
})
