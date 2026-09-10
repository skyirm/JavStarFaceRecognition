<script>
  import ImagePicker from './ImagePicker.svelte'
  import { api } from './api.js'

  let file = $state(null)
  let hits = $state([])
  let busy = $state(false)
  let error = $state('')
  let canvas = $state(null)

  async function run() {
    if (!file || busy) return
    busy = true
    error = ''
    hits = []
    try {
      hits = await api.recognize(file)
      await draw()
    } catch (e) {
      error = e.message
    }
    busy = false
  }

  async function draw() {
    const bitmap = await createImageBitmap(file)
    canvas.width = bitmap.width
    canvas.height = bitmap.height
    const ctx = canvas.getContext('2d')
    ctx.drawImage(bitmap, 0, 0)
    const line = Math.max(2, Math.round(bitmap.width / 300))
    const font = Math.max(16, Math.round(bitmap.height / 28))
    ctx.font = `bold ${font}px system-ui, sans-serif`
    ctx.textBaseline = 'bottom'
    for (const hit of hits) {
      const [x1, y1, x2, y2] = hit.bbox
      const color = hit.name === 'Unknown' ? '#f59e0b' : '#ef4444'
      ctx.strokeStyle = color
      ctx.lineWidth = line
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
      const label = hit.name === 'Unknown' ? 'Unknown' : `${hit.name} ${hit.similarity.toFixed(2)}`
      const ty = y1 - 4 < font ? y1 + font + 4 : y1 - 4
      ctx.lineWidth = Math.max(3, font / 6)
      ctx.strokeStyle = '#fff'
      ctx.strokeText(label, x1, ty)
      ctx.fillStyle = color
      ctx.fillText(label, x1, ty)
    }
    bitmap.close()
  }
</script>

<div class="card">
  <ImagePicker bind:file label="上传图片" />
  <button class="primary" onclick={run} disabled={!file || busy}>
    {busy ? '识别中…' : '开始识别'}
  </button>
  {#if error}
    <div class="msg err">{error}</div>
  {/if}
  {#if hits.length}
    <canvas bind:this={canvas} class="result"></canvas>
    <ul class="hits">
      {#each hits as hit}
        <li>
          {hit.name === 'Unknown' ? '❓ Unknown' : `👤 ${hit.name}`}
          <span class="sim">相似度 {hit.similarity.toFixed(4)}</span>
        </li>
      {/each}
    </ul>
  {/if}
</div>
