<script>
  import ImagePicker from './ImagePicker.svelte'
  import AdminToken from './AdminToken.svelte'
  import { api } from './api.js'

  let name = $state('')
  let files = $state([])
  let suggestions = $state([])
  let busy = $state(false)
  let msg = $state('')
  let ok = $state(false)
  let needToken = $state(false)
  let progress = $state('')

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
    if (!files.length) {
      msg = '请选择图片'
      ok = false
      return
    }
    busy = true
    msg = ''
    progress = ''
    const total = files.length
    let done = 0
    let count = 0
    const failed = []
    const succeeded = new Set()
    let halted = false
    for (const f of files) {
      progress = `${done + 1}/${total}`
      try {
        const r = await api.uploadFace(name.trim(), f)
        succeeded.add(f)
        done++
        count = r.count
      } catch (e) {
        if (e.status === 401) {
          needToken = true
          halted = true
          break
        }
        failed.push(`${f.name}：${e.message}`)
      }
    }
    files = files.filter((f) => !succeeded.has(f))
    progress = ''
    busy = false
    if (halted) {
      msg = `已成功 ${done} 张后需要管理员令牌，剩余图片未尝试`
      ok = done > 0
    } else if (failed.length) {
      msg = `成功 ${done} 张，失败 ${failed.length} 张：${failed.join('；')}`
      ok = done > 0
    } else {
      msg = `已上传 ${done} 张「${name.trim()}」，当前共 ${count} 张注册图`
      ok = true
      suggestions = []
    }
  }
</script>

<div class="card">
  <div>
    <label class="field-label" for="name-input">输入名字</label>
    <input
      id="name-input"
      type="text"
      class="field-input"
      bind:value={name}
      oninput={onname}
      placeholder="输入名字，会提示已有相似名字"
    />
    {#if suggestions.length}
      <div class="mt-2 flex items-center gap-2 flex-wrap">
        <span class="text-xs text-muted">推荐使用：</span>
        {#each suggestions as s}
          <button
            class="btn-chip"
            onclick={() => {
              name = s
              suggestions = []
            }}
          >
            {s}
          </button>
        {/each}
      </div>
    {/if}
  </div>
  <ImagePicker bind:files multiple label="上传人脸图片（可多选，需为单人照）" />
  <button class="btn-primary self-start" onclick={submit} disabled={busy}>
    {busy ? (progress ? `上传中 ${progress}…` : '上传中…') : '上传'}
  </button>
  {#if needToken}
    <AdminToken onsave={() => { needToken = false }} />
  {/if}
  {#if msg}
    <div class="msg {ok ? 'msg-ok' : 'msg-err'}">{msg}</div>
  {/if}
</div>
