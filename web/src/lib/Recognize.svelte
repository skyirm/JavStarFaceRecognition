<script>
  import ImagePicker from './ImagePicker.svelte'
  import { api } from './api.js'

  let file = $state(null)
  let recognizedFile = null
  let hits = $state([])
  let busy = $state(false)
  let error = $state('')
  let notice = $state('')
  let canvas = $state(null)

  let regFor = $state(-1)
  let regName = $state('')
  let regSuggestions = $state([])
  let regBusy = $state(false)
  let regMsg = $state('')
  let regOk = $state(false)
  let regTimer

  async function run() {
    if (!file || busy) return
    busy = true
    error = ''
    notice = ''
    hits = []
    regFor = -1
    try {
      hits = await api.recognize(file)
      recognizedFile = file
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

  function toggleRegister(i) {
    if (regFor === i) {
      regFor = -1
      return
    }
    regFor = i
    regName = ''
    regSuggestions = []
    regMsg = ''
    regOk = false
  }

  function onRegName() {
    clearTimeout(regTimer)
    if (!regName.trim()) {
      regSuggestions = []
      return
    }
    regTimer = setTimeout(async () => {
      try {
        regSuggestions = (await api.suggest(regName.trim())).suggestions
      } catch {
        regSuggestions = []
      }
    }, 250)
  }

  async function cropFace(hit) {
    const bitmap = await createImageBitmap(recognizedFile)
    const [x1, y1, x2, y2] = hit.bbox
    const pad = 0.25
    const bw = x2 - x1
    const bh = y2 - y1
    const sx = Math.max(0, Math.floor(x1 - bw * pad))
    const sy = Math.max(0, Math.floor(y1 - bh * pad))
    const sw = Math.min(bitmap.width - sx, Math.ceil(bw * (1 + pad * 2)))
    const sh = Math.min(bitmap.height - sy, Math.ceil(bh * (1 + pad * 2)))
    if (sw <= 0 || sh <= 0) {
      bitmap.close()
      throw new Error('人脸区域无效')
    }
    const c = document.createElement('canvas')
    c.width = sw
    c.height = sh
    c.getContext('2d').drawImage(bitmap, sx, sy, sw, sh, 0, 0, sw, sh)
    bitmap.close()
    return new Promise((resolve, reject) =>
      c.toBlob(
        (b) =>
          b
            ? resolve(new File([b], 'face.jpg', { type: 'image/jpeg' }))
            : reject(new Error('人脸裁剪失败')),
        'image/jpeg',
        0.95
      )
    )
  }

  async function submitRegister(hit) {
    if (regBusy) return
    const name = regName.trim()
    if (!name) {
      regMsg = '名字不能为空'
      regOk = false
      return
    }
    if (file !== recognizedFile) {
      regMsg = '图片已更换，请重新识别'
      regOk = false
      return
    }
    regBusy = true
    regMsg = ''
    try {
      const cropped = await cropFace(hit)
      const r = await api.uploadFace(name, cropped)
      hit.registered = name
      notice = `已注册「${name}」，当前共 ${r.count} 张注册图`
      regFor = -1
    } catch (e) {
      regMsg = e.message
      regOk = false
    }
    regBusy = false
  }
</script>

<div class="card">
  <ImagePicker bind:file label="上传图片" />
  <button class="btn-primary w-full sm:w-auto sm:self-start" onclick={run} disabled={!file || busy}>
    {busy ? '识别中…' : '开始识别'}
  </button>
  {#if error}
    <div class="msg msg-err">{error}</div>
  {/if}
  {#if notice}
    <div class="msg msg-ok">{notice}</div>
  {/if}
  {#if hits.length}
    <canvas bind:this={canvas} class="w-full rounded-lg border border-hairline-dark"></canvas>
    <ul class="rounded-lg border border-hairline-dark divide-y divide-hairline-dark">
      {#each hits as hit, i (i)}
        <li class="px-4 py-3 text-sm">
          <div class="flex items-center gap-2.5">
            <span
              class="w-1.5 h-1.5 rounded-full shrink-0
                {hit.name === 'Unknown' && !hit.registered ? 'bg-trading-down' : 'bg-trading-up'}"
            ></span>
            {#if hit.registered}
              <span class="text-white font-medium">{hit.registered}</span>
              <span
                class="text-xs px-2 py-0.5 rounded-full border border-trading-up/40 text-trading-up"
              >已注册</span>
            {:else if hit.name === 'Unknown'}
              <span class="text-trading-down font-medium">Unknown</span>
            {:else}
              <span class="text-white font-medium">{hit.name}</span>
            {/if}
            {#if !hit.registered}
              <span
                class="ml-auto font-num text-sm {hit.name === 'Unknown'
                  ? 'text-muted'
                  : 'text-trading-up'}"
              >
                {hit.name === 'Unknown' ? '—' : `▲ ${hit.similarity.toFixed(4)}`}
              </span>
            {/if}
          </div>
          {#if candidateLine(hit)}
            <div class="mt-1 text-xs text-muted font-num break-words">{candidateLine(hit)}</div>
          {/if}
          {#if hit.name === 'Unknown' && !hit.registered}
            {#if regFor === i}
              <div class="mt-3 flex flex-col gap-2">
                <div class="flex flex-wrap gap-2">
                  <input
                    type="text"
                    class="field-input flex-1 min-w-40"
                    bind:value={regName}
                    oninput={onRegName}
                    placeholder="输入此人的名字"
                    onkeydown={(e) => {
                      if (e.key === 'Enter' && !regBusy) submitRegister(hit)
                    }}
                  />
                  <button class="btn-primary" onclick={() => submitRegister(hit)} disabled={regBusy}>
                    {regBusy ? '注册中…' : '确认注册'}
                  </button>
                  <button class="btn-secondary" onclick={() => (regFor = -1)} disabled={regBusy}>
                    取消
                  </button>
                </div>
                {#if regSuggestions.length}
                  <div class="flex items-center gap-2 flex-wrap">
                    <span class="text-xs text-muted">已有名字，推荐使用：</span>
                    {#each regSuggestions as s (s)}
                      <button
                        class="btn-chip"
                        onclick={() => {
                          regName = s
                          regSuggestions = []
                        }}
                      >
                        {s}
                      </button>
                    {/each}
                  </div>
                {/if}
                {#if regMsg}
                  <div class="msg {regOk ? 'msg-ok' : 'msg-err'}">{regMsg}</div>
                {/if}
              </div>
            {:else}
              <button class="btn-chip mt-2" onclick={() => toggleRegister(i)}>＋ 注册此人</button>
            {/if}
          {/if}
        </li>
      {/each}
    </ul>
  {/if}
</div>
