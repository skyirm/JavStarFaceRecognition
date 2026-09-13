<script module>
  // ---- clipboard paste routing (shared across all instances) ----
  // Paste goes to the picker the user interacted with last; before any
  // interaction it falls back to the first mounted picker on the page.
  const acceptors = new Set()
  let active = null

  function fileFromClipboard(dt) {
    if (!dt) return null
    for (const item of dt.items) {
      if (item.kind === 'file' && item.type.startsWith('image/')) {
        const raw = item.getAsFile()
        if (raw) {
          const ext = raw.type.split('/')[1] || 'png'
          return new File([raw], `clipboard_${Date.now()}.${ext}`, { type: raw.type })
        }
      }
    }
    return null
  }

  if (typeof window !== 'undefined' && !window.__facePasteBound) {
    window.__facePasteBound = true
    window.addEventListener('paste', (e) => {
      const target = active ?? [...acceptors][0]
      if (!target) return
      const file = fileFromClipboard(e.clipboardData)
      if (!file) return
      e.preventDefault()
      target(file)
    })
  }
</script>

<script>
  let {
    label = '选择图片',
    file = $bindable(null),
    multiple = false,
    files = $bindable([])
  } = $props()

  let url = $state('')
  let urls = new Map()
  let dragging = $state(false)

  function urlFor(f) {
    if (!urls.has(f)) urls.set(f, URL.createObjectURL(f))
    return urls.get(f)
  }

  function setFile(f) {
    if (!f || !f.type.startsWith('image/')) return
    if (url) URL.revokeObjectURL(url)
    file = f
    url = URL.createObjectURL(f)
  }

  function isSameFile(a, b) {
    return a.name === b.name && a.size === b.size && a.lastModified === b.lastModified
  }

  function addFiles(list) {
    const merged = [...files]
    for (const f of list) {
      if (!f.type.startsWith('image/')) continue
      if (merged.some((x) => isSameFile(x, f))) continue
      merged.push(f)
    }
    files = merged
  }

  function removeAt(i) {
    files = files.filter((_, idx) => idx !== i)
  }

  $effect(() => {
    if (!multiple) return
    const current = [...files]
    for (const [f, u] of [...urls]) {
      if (!current.includes(f)) {
        URL.revokeObjectURL(u)
        urls.delete(f)
      }
    }
  })

  function onpick(e) {
    if (multiple) {
      addFiles(e.target.files)
      e.target.value = ''
    } else {
      setFile(e.target.files[0] ?? null)
    }
  }

  function ondrop(e) {
    e.preventDefault()
    dragging = false
    if (multiple) addFiles(e.dataTransfer.files)
    else setFile(e.dataTransfer.files[0] ?? null)
  }

  function ondragover(e) {
    e.preventDefault()
    dragging = true
  }

  function ondragleave(e) {
    if (!e.currentTarget.contains(e.relatedTarget)) dragging = false
  }

  function accept(f) {
    if (multiple) addFiles([f])
    else setFile(f)
  }

  function onpointerdown() {
    active = accept
  }

  $effect(() => {
    acceptors.add(accept)
    return () => {
      acceptors.delete(accept)
      if (active === accept) active = null
    }
  })

  $effect(() => {
    if (!file && url) {
      URL.revokeObjectURL(url)
      url = ''
    }
  })
</script>

<div class="min-w-0">
  <span class="field-label">{label}</span>
  <label
    class="flex flex-col items-center justify-center gap-1.5 rounded-lg border border-dashed
      border-hairline-dark hover:border-primary/60 cursor-pointer px-4 py-6
      bg-surface-card/40 transition-colors text-center
      {dragging ? 'border-solid border-primary bg-primary/5' : ''}"
    ondragover={ondragover}
    ondragleave={ondragleave}
    ondrop={ondrop}
    onpointerdown={onpointerdown}
  >
    <input
      type="file"
      accept="image/jpeg,image/png,image/webp"
      {multiple}
      class="sr-only"
      onchange={onpick}
    />
    {#if multiple}
      {#if files.length}
        <div class="grid grid-cols-3 sm:grid-cols-5 gap-2 w-full">
          {#each files as f, i (i)}
            <div class="relative">
              <img
                src={urlFor(f)}
                alt={f.name}
                class="w-full h-20 object-cover rounded-md border border-hairline-dark"
              />
              <button
                type="button"
                class="absolute top-0 right-0 w-7 h-7 rounded-full bg-black/60
                  text-white text-sm leading-none cursor-pointer flex items-center justify-center"
                title="移除"
                onclick={(e) => {
                  e.preventDefault()
                  removeAt(i)
                }}
              >×</button>
            </div>
          {/each}
        </div>
        <span class="text-xs text-muted">点击继续添加、拖入或 Ctrl+V 粘贴，共 {files.length} 张</span>
      {:else}
        <svg viewBox="0 0 24 24" class="w-7 h-7 fill-muted-strong" aria-hidden="true">
          <path
            d="M11 16V7.8l-3.6 3.6L6 10l6-6 6 6-1.4 1.4L13 7.8V16h-2Zm-6 4v-4h2v2h10v-2h2v4H5Z"
          />
        </svg>
        <span class="text-sm text-body">点击选择、拖入或 Ctrl+V 粘贴（可多选）</span>
        <span class="text-xs text-muted">支持 JPG / PNG / WebP，≤ 50MB</span>
      {/if}
    {:else if url}
      <img src={url} alt="预览" class="max-h-48 rounded-md border border-hairline-dark" />
      <span class="text-xs text-muted">点击更换、拖入或 Ctrl+V 粘贴</span>
    {:else}
      <svg viewBox="0 0 24 24" class="w-7 h-7 fill-muted-strong" aria-hidden="true">
        <path
          d="M11 16V7.8l-3.6 3.6L6 10l6-6 6 6-1.4 1.4L13 7.8V16h-2Zm-6 4v-4h2v2h10v-2h2v4H5Z"
        />
      </svg>
      <span class="text-sm text-body">点击选择、拖入或 Ctrl+V 粘贴</span>
      <span class="text-xs text-muted">支持 JPG / PNG / WebP，≤ 50MB</span>
    {/if}
  </label>
  {#if !multiple && file}
    <div class="mt-1.5 text-xs text-muted-strong truncate" title={file.name}>{file.name}</div>
  {/if}
</div>
