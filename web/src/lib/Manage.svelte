<script>
  import { api } from './api.js'

  let persons = $state([])
  let query = $state('')
  let loading = $state(false)
  let msg = $state('')

  let filtered = $derived(
    persons.filter(
      (p) => !query.trim() || p.name.toLowerCase().includes(query.trim().toLowerCase())
    )
  )

  $effect(() => {
    refresh()
  })

  async function refresh() {
    loading = true
    msg = ''
    try {
      persons = await api.listFaces()
    } catch (e) {
      msg = e.message
    }
    loading = false
  }

  async function del(name) {
    if (!confirm(`确认删除「${name}」的全部注册向量？`)) return
    try {
      const r = await api.deleteFace(name)
      await refresh()
      msg = `已删除 ${r.deleted}（${r.vectors} 条向量）`
    } catch (e) {
      msg = e.message
    }
  }
</script>

<div class="card">
  <div>
    <label class="field-label" for="search">搜索</label>
    <input id="search" type="text" bind:value={query} placeholder="按名字过滤" />
  </div>
  {#if msg}
    <div class="msg ok">{msg}</div>
  {/if}
  {#if loading}
    <div>加载中…</div>
  {:else if filtered.length === 0}
    <div>人脸库为空{query ? '，无匹配结果' : ''}</div>
  {:else}
    <table class="persons">
      <thead>
        <tr><th>名字</th><th>向量数</th><th></th></tr>
      </thead>
      <tbody>
        {#each filtered as p (p.name)}
          <tr>
            <td>{p.name}</td>
            <td>{p.count}</td>
            <td><button class="danger" onclick={() => del(p.name)}>删除</button></td>
          </tr>
        {/each}
      </tbody>
    </table>
    <div class="field-label">共 {filtered.length} 人</div>
  {/if}
</div>
