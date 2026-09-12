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
      const color = hit.name === 'Unknown' ? '#f6465d' : '#0ecb81'
      ctx.strokeStyle = color
      ctx.lineWidth = line
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
      const label = hit.name === 'Unknown' ? 'Unknown' : `${hit.name} ${hit.similarity.toFixed(2)}`
      const ty = y1 - 4 < font ? y1 + font + 4 : y1 - 4
      ctx.lineWidth = Math.max(3, font / 6)
      ctx.strokeStyle = '#0b0e11'
      ctx.strokeText(label, x1, ty)
      ctx.fillStyle = color
      ctx.fillText(label, x1, ty)
    }
    bitmap.close()
  }
  function candidateLine(hit) {
    if (!hit.candidates || hit.candidates.length < 2) return ''
    const list =
      hit.name === 'Unknown' ? hit.candidates : hit.candidates.slice(1)
    if (!list.length) return ''
    const text = list
      .map((c) => `${c.name} ${c.similarity.toFixed(4)}`)
      .join(' · ')
    return hit.name === 'Unknown' ? `最接近：${text}` : `候选：${text}`
  }
</script>

<div class="card">
  <ImagePicker bind:file label="上传图片" />
  <button class="btn-primary self-start" onclick={run} disabled={!file || busy}>
    {busy ? '识别中…' : '开始识别'}
  </button>
  {#if error}
    <div class="msg msg-err">{error}</div>
  {/if}
  {#if hits.length}
    <canvas bind:this={canvas} class="w-full rounded-lg border border-hairline-dark"></canvas>
    <ul class="rounded-lg border border-hairline-dark divide-y divide-hairline-dark">
      {#each hits as hit}
        <li class="px-4 py-3 text-sm">
          <div class="flex items-center gap-2.5">
            <span
              class="w-1.5 h-1.5 rounded-full shrink-0
                {hit.name === 'Unknown' ? 'bg-trading-down' : 'bg-trading-up'}"
            ></span>
            {#if hit.name === 'Unknown'}
              <span class="text-trading-down font-medium">Unknown</span>
            {:else}
              <span class="text-white font-medium">{hit.name}</span>
            {/if}
            <span
              class="ml-auto font-num text-sm {hit.name === 'Unknown'
                ? 'text-muted'
                : 'text-trading-up'}"
            >
              {hit.name === 'Unknown' ? '—' : `▲ ${hit.similarity.toFixed(4)}`}
            </span>
          </div>
          {#if candidateLine(hit)}
            <div class="mt-1 text-xs text-muted font-num">{candidateLine(hit)}</div>
          {/if}
        </li>
      {/each}
    </ul>
  {/if}
</div>
