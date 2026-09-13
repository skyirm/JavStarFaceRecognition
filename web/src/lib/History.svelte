<script>
  import { api } from './api.js'
  import AdminToken from './AdminToken.svelte'

  let items = $state([])
  let total = $state(0)
  let page = $state(1)
  const pageSize = 50
  let unknownOnly = $state(false)
  let query = $state('')
  let currentQ = ''
  let searchTimer
  let loading = $state(false)
  let msg = $state('')
  let msgOk = $state(true)
  let needToken = $state(false)

  let totalPages = $derived(Math.max(1, Math.ceil(total / pageSize)))

  $effect(() => {
    refresh(1)
  })

  async function refresh(p = 1) {
    loading = true
    msg = ''
    try {
      const r = await api.history({
        q: currentQ,
        unknown: unknownOnly,
        page: p,
        page_size: pageSize
      })
      if (r.items.length === 0 && p > 1) {
        loading = false
        return refresh(p - 1)
      }
      items = r.items
      total = r.total
      page = r.page
      needToken = false
    } catch (e) {
      if (e.status === 401) needToken = true
      msg = e.message
      msgOk = false
    }
    loading = false
  }

  function onsearch() {
    clearTimeout(searchTimer)
    searchTimer = setTimeout(() => {
      currentQ = query
      refresh(1)
    }, 250)
  }

  function toggleUnknown() {
    unknownOnly = !unknownOnly
    refresh(1)
  }

  function gotoPage(p) {
    if (p >= 1 && p <= totalPages && p !== page) refresh(p)
  }

  function fmtTime(ts) {
    return ts.replace('T', ' ')
  }

  async function delRecord(id) {
    if (!confirm('确认删除这条识别记录？')) return
    try {
      await api.deleteHistory(id)
      await refresh()
      notify('已删除记录')
    } catch (e) {
      notify(e.message, false)
    }
  }

  async function clearAll() {
    if (!confirm(`确认清空全部 ${total} 条识别记录？此操作不可恢复`)) return
    try {
      const r = await api.clearHistory()
      await refresh(1)
      notify(`已清空 ${r.deleted} 条记录`)
    } catch (e) {
      notify(e.message, false)
    }
  }

  function notify(text, ok = true) {
    msg = text
    msgOk = ok
  }
</script>

<div class="card">
  {#if needToken}
    <div class="msg msg-err">此页面需要管理员令牌（服务端已设置 ADMIN_TOKEN）</div>
    <AdminToken onsave={() => refresh(1)} />
  {/if}
  <div class="flex flex-wrap items-end gap-2">
    <div class="flex-1 min-w-45">
      <label class="field-label" for="hsearch">搜索</label>
      <input
        id="hsearch"
        type="text"
        class="field-input"
        bind:value={query}
        oninput={onsearch}
        placeholder="按识别出的人名过滤（Unknown 用右侧开关）"
      />
    </div>
    <button class="btn-chip h-11 sm:h-10" class:ring-1={unknownOnly} class:ring-primary={unknownOnly} onclick={toggleUnknown}>
      {unknownOnly ? '✓ 仅看 Unknown' : '仅看 Unknown'}
    </button>
    {#if total > 0}
      <button class="btn-danger h-11 sm:h-10" onclick={clearAll}>清空历史</button>
    {/if}
  </div>
  {#if msg}
    <div class="msg {msgOk ? 'msg-ok' : 'msg-err'}">{msg}</div>
  {/if}
  {#if loading}
    <div class="text-sm text-muted py-6 text-center">加载中…</div>
  {:else if items.length === 0}
    <div class="text-sm text-muted py-6 text-center">
      暂无识别记录{currentQ || unknownOnly ? '，无匹配结果' : ''}
    </div>
  {:else}
    <ul class="rounded-lg border border-hairline-dark divide-y divide-hairline-dark">
      {#each items as it (it.id)}
        <li class="px-4 py-3 flex items-center gap-3">
          {#if it.thumb}
            <img
              src={it.thumb}
              alt="识别记录 #{it.id}"
              class="w-14 h-14 rounded-md border border-hairline-dark object-cover shrink-0"
            />
          {:else}
            <div
              class="w-14 h-14 rounded-md border border-dashed border-hairline-dark
                flex items-center justify-center text-[10px] text-muted shrink-0"
            >
              无图
            </div>
          {/if}
          <div class="min-w-0 flex-1">
            <div class="flex items-center gap-2 flex-wrap">
              {#if it.name}
                <span class="text-white font-medium">{it.name}</span>
              {:else}
                <span class="text-trading-down font-medium">Unknown</span>
              {/if}
              <span class="font-num text-xs {it.name ? 'text-trading-up' : 'text-muted'}">
                {it.name ? `▲ ${Number(it.similarity).toFixed(4)}` : '—'}
              </span>
            </div>
            <div class="mt-0.5 text-xs text-muted font-num">{fmtTime(it.ts)}</div>
          </div>
          <button class="btn-danger shrink-0" onclick={() => delRecord(it.id)}>删除</button>
        </li>
      {/each}
    </ul>
    <div class="flex items-center justify-between gap-3 text-xs text-muted">
      <span>共 {total} 条</span>
      {#if totalPages > 1}
        <div class="flex items-center gap-2">
          <button class="btn-chip" disabled={page <= 1} onclick={() => gotoPage(page - 1)}>
            上一页
          </button>
          <span class="font-num">第 {page} / {totalPages} 页</span>
          <button class="btn-chip" disabled={page >= totalPages} onclick={() => gotoPage(page + 1)}>
            下一页
          </button>
        </div>
      {/if}
    </div>
  {/if}
</div>
