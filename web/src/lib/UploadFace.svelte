<script>
  import ImagePicker from './ImagePicker.svelte'
  import { api } from './api.js'

  let name = $state('')
  let file = $state(null)
  let suggestions = $state([])
  let busy = $state(false)
  let msg = $state('')
  let ok = $state(false)

  let timer

  function onname() {
    clearTimeout(timer)
    if (!name.trim()) {
      suggestions = []
      return
    }
    timer = setTimeout(async () => {
      try {
        suggestions = (await api.suggest(name.trim())).suggestions
      } catch {
        suggestions = []
      }
    }, 250)
  }

  async function submit() {
    if (busy) return
    if (!name.trim()) {
      msg = '名字不能为空'
      ok = false
      return
    }
    if (!file) {
      msg = '请选择图片'
      ok = false
      return
    }
    busy = true
    msg = ''
    try {
      const r = await api.uploadFace(name.trim(), file)
      msg = `${r.name} 上传成功，当前共 ${r.count} 张注册图`
      ok = true
      file = null
      suggestions = []
    } catch (e) {
      msg = e.message
      ok = false
    }
    busy = false
  }
</script>

<div class="card">
  <div>
    <label class="field-label" for="name-input">输入名字</label>
    <input
      id="name-input"
      type="text"
      bind:value={name}
      oninput={onname}
      placeholder="输入名字，会提示已有相似名字"
    />
    {#if suggestions.length}
      <div class="suggestions" style="margin-top: 8px">
        <span class="field-label">推荐使用：</span>
        {#each suggestions as s}
          <button onclick={() => { name = s; suggestions = [] }}>{s}</button>
        {/each}
      </div>
    {/if}
  </div>
  <ImagePicker bind:file label="上传人脸图片（需为单人照）" />
  <button class="primary" onclick={submit} disabled={busy}>
    {busy ? '上传中…' : '上传'}
  </button>
  {#if msg}
    <div class="msg {ok ? 'ok' : 'err'}">{msg}</div>
  {/if}
</div>
