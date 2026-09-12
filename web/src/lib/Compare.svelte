<script>
  import ImagePicker from './ImagePicker.svelte'
  import { api } from './api.js'

  let file1 = $state(null)
  let file2 = $state(null)
  let similarity = $state(null)
  let busy = $state(false)
  let error = $state('')

  async function run() {
    if (busy) return
    if (!file1 || !file2) {
      error = '请选择两张图片'
      similarity = null
      return
    }
    busy = true
    error = ''
    similarity = null
    try {
      similarity = (await api.compare(file1, file2)).similarity
    } catch (e) {
      error = e.message
    }
    busy = false
  }
</script>

<div class="card">
  <div class="grid gap-4 md:grid-cols-2">
    <ImagePicker bind:file={file1} label="图片 1" />
    <ImagePicker bind:file={file2} label="图片 2" />
  </div>
  <button class="btn-primary self-start" onclick={run} disabled={busy || !file1 || !file2}>
    {busy ? '比对中…' : '开始比对'}
  </button>
  {#if error}
    <div class="msg msg-err">{error}</div>
  {/if}
  {#if similarity !== null}
    <div class="rounded-lg border border-hairline-dark p-6">
      <div class="flex items-baseline gap-4">
        <span class="text-xs font-medium text-muted">相似度</span>
        <span
          class="font-num text-4xl font-bold tracking-tight
            {similarity >= 0.75 ? 'text-trading-up' : 'text-trading-down'}"
        >
          {(similarity * 100).toFixed(1)}%
        </span>
        <span class="ml-auto text-xs text-muted">
          {similarity >= 0.75 ? '很可能是同一人' : '可能不是同一人'}
        </span>
      </div>
      <div class="mt-4 h-1.5 rounded-full bg-surface-elevated overflow-hidden">
        <div
          class="h-full rounded-full {similarity >= 0.75 ? 'bg-trading-up' : 'bg-trading-down'}"
          style="width: {Math.min(100, similarity * 100)}%"
        ></div>
      </div>
    </div>
  {/if}
</div>
