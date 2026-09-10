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
  <div class="row">
    <ImagePicker bind:file={file1} label="图片 1" />
    <ImagePicker bind:file={file2} label="图片 2" />
  </div>
  <button class="primary" onclick={run} disabled={busy || !file1 || !file2}>
    {busy ? '比对中…' : '比较'}
  </button>
  {#if error}
    <div class="msg err">{error}</div>
  {/if}
  {#if similarity !== null}
    <div>
      <div class="field-label">比对结果</div>
      <div class="similarity-big">{(similarity * 100).toFixed(1)}%</div>
    </div>
  {/if}
</div>
