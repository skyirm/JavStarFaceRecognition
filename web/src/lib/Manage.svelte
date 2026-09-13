<script>
  import { api } from './api.js'
  import AdminToken from './AdminToken.svelte'

  let persons = $state([])
  let total = $state(0)
  let page = $state(1)
  const pageSize = 50
  let query = $state('')
  let currentQ = ''
  let searchTimer
  let loading = $state(false)
  let msg = $state('')
  let msgOk = $state(true)
  let aliasTarget = $state(null)
  let aliasInput = $state('')
  let needToken = $state(false)
  let expanded = $state(null)
  let thumbs = $state({})
  let issues = $state(null)
  let issuesLoading = $state(false)
  let importing = $state(false)
  let importInput

  let totalPages = $derived(Math.max(1, Math.ceil(total / pageSize)))

  $effect(() => {
    refresh(1)
  })

  async function refresh(p = 1) {
    loading = true
    msg = ''
    try {
      const r = await api.listFaces({ q: currentQ, page: p, page_size: pageSize })
      if (r.items.length === 0 && p > 1) {
        loading = false
        return refresh(p - 1)
      }
      persons = r.items
      total = r.total
      page = r.page
      needToken = false
    } catch (e) {
      if (e.status === 401) needToken = true
      msg = e.message
      msgOk = false
    }
    thumbs = {}
    expanded = null
    loading = false
  }

  function onsearch() {
    clearTimeout(searchTimer)
    searchTimer = setTimeout(() => {
      currentQ = query
      refresh(1)
    }, 250)
  }

  function gotoPage(p) {
    if (p >= 1 && p <= totalPages && p !== page) refresh(p)
  }

  async function runIssues() {
    issuesLoading = true
    try {
      issues = await api.faceIssues()
      needToken = false
    } catch (e) {
      if (e.status === 401) needToken = true
      notify(e.message, false)
    }
    issuesLoading = false
  }

  async function mergePair(pair, target) {
    const source = target === pair.a_name ? pair.b_name : pair.a_name
    if (!confirm(`将「${source}」合并到「${target}」？（${source} 会成为其别名）`)) return
    try {
      await api.mergeFace(source, target)
      await refresh()
      issues = null
      notify(`已将「${source}」合并到「${target}」`)
    } catch (e) {
      notify(e.message, false)
    }
  }

  async function delOutlier(o) {
    if (!confirm(`确认删除「${o.name}」的疑似离群向量 #${o.id}？（与本人均值相似度 ${o.sim_to_mean.toFixed(4)}）`)) return
    try {
      const r = await api.deleteVector(o.name, o.id)
      await refresh()
      if (r.person_gone) issues = null
      else await runIssues()
      notify(`已删除「${o.name}」的向量 #${o.id}`)
    } catch (e) {
      notify(e.message, false)
    }
  }

  async function doExport() {
    try {
      const blob = await api.exportLibrary()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `face_library_${new Date().toISOString().slice(0, 10)}.zip`
      a.click()
      URL.revokeObjectURL(url)
      notify('已导出人脸库 zip（含向量、别名、预览图）')
    } catch (e) {
      if (e.status === 401) needToken = true
      notify(e.message, false)
    }
  }

  async function onImportFile(e) {
    const f = e.target.files[0]
    e.target.value = ''
    if (!f) return
    if (!confirm(`确认导入「${f.name}」？当前人脸库（向量与别名）将被完全替换`)) return
    importing = true
    try {
      const r = await api.importLibrary(f)
      await refresh(1)
      issues = null
      notify(`导入成功：${r.persons} 人 / ${r.vectors} 条向量 / ${r.aliases} 条别名`)
    } catch (e2) {
      if (e2.status === 401) needToken = true
      notify(e2.message, false)
    }
    importing = false
  }

  async function toggleThumbs(name) {
    if (expanded === name) {
      expanded = null
      return
    }
    expanded = name
    if (thumbs[name]) return
    thumbs = { ...thumbs, [name]: { loading: true, items: [] } }
    try {
      const items = await api.faceThumbs(name)
      thumbs = { ...thumbs, [name]: { loading: false, items } }
    } catch (e) {
      if (e.status === 401) needToken = true
      thumbs = { ...thumbs, [name]: { loading: false, items: [] } }
      notify(e.message, false)
    }
  }

  function notify(text, ok = true) {
    msg = text
    msgOk = ok
  }

  async function del(name) {
    if (!confirm(`确认删除「${name}」的全部注册向量？`)) return
    try {
      const r = await api.deleteFace(name)
      await refresh()
      notify(`已删除 ${r.deleted}（${r.vectors} 条向量）`)
    } catch (e) {
      notify(e.message, false)
    }
  }

  function openAliasInput(name) {
    aliasTarget = name
    aliasInput = ''
  }

  function closeAliasInput() {
    aliasTarget = null
    aliasInput = ''
  }

  async function saveAlias(event) {
    if (event && event.key && event.key !== 'Enter') return
    const name = aliasTarget
    const alias = aliasInput.trim()
    if (!alias) return
    try {
      const r = await api.addAlias(name, alias)
      await refresh()
      closeAliasInput()
      const list = r.aliases.join('、')
      notify(`「${r.name}」的别名：${list}`)
    } catch (e) {
      notify(e.message, false)
    }
  }

  async function removeAlias(name, alias) {
    if (!confirm(`确认删除别名「${alias}」？`)) return
    try {
      await api.removeAlias(name, alias)
      await refresh()
      notify(`已删除别名「${alias}」`)
    } catch (e) {
      notify(e.message, false)
    }
  }

  async function delVector(p, id) {
    if (!confirm(`确认删除「${p.name}」的一条注册向量？此操作不可恢复`)) return
    try {
      const r = await api.deleteVector(p.name, id)
      if (r.person_gone) {
        await refresh()
        notify(`已删除「${p.name}」最后一条向量，该人已从库中移除`)
      } else {
        thumbs = {
          ...thumbs,
          [p.name]: {
            ...thumbs[p.name],
            items: thumbs[p.name].items.filter((t) => t.id !== id)
          }
        }
        persons = persons.map((x) =>
          x.name === p.name ? { ...x, count: r.remaining } : x
        )
        notify(`已删除 1 条向量，「${p.name}」剩余 ${r.remaining} 条`)
      }
    } catch (e) {
      if (e.status === 401) needToken = true
      notify(e.message, false)
    }
  }

  async function merge(p) {
    const target = prompt(
      `将「${p.name}」合并到哪个人？（输入目标主名，其向量会全部移过去，「${p.name}」变成别名）`
    )
    if (target === null) return
    const t = target.trim()
    if (!t) return
    try {
      const r = await api.mergeFace(p.name, t)
      await refresh()
      notify(`已将「${p.name}」合并到「${r.name}」`)
    } catch (e) {
      notify(e.message, false)
    }
  }
</script>

{#snippet aliasChips(p)}
  {#if p.aliases.length}
    <div class="flex flex-wrap gap-1.5">
      {#each p.aliases as a (a)}
        <span
          class="inline-flex items-center gap-1 h-7 pl-3 pr-1 rounded-full
            border border-hairline-dark bg-surface-elevated text-xs text-body"
        >
          {a}
          <button
            class="w-5 h-5 rounded-full text-muted hover:text-trading-down
              cursor-pointer leading-none"
            title="删除别名"
            onclick={() => removeAlias(p.name, a)}
          >×</button>
        </span>
      {/each}
    </div>
  {/if}
{/snippet}

{#snippet aliasEditor(p)}
  {#if aliasTarget === p.name}
    <div class="flex flex-wrap gap-2 mt-1">
      <input
        type="text"
        class="field-input !h-8 !py-0 text-xs flex-1 min-w-[9rem]"
        bind:value={aliasInput}
        onkeydown={saveAlias}
        placeholder="输入别名（曾用名/艺名）"
      />
      <button class="btn-chip" onclick={() => saveAlias()}>保存</button>
      <button class="btn-chip" onclick={closeAliasInput}>取消</button>
    </div>
  {/if}
{/snippet}

{#snippet thumbGrid(p)}
  {#if expanded === p.name}
    <div class="mt-2 flex flex-wrap gap-1.5">
      {#if thumbs[p.name]?.loading}
        <span class="text-xs text-muted py-2">加载预览…</span>
      {:else if (thumbs[p.name]?.items ?? []).length === 0}
        <span class="text-xs text-muted py-2">无向量</span>
      {:else}
        {#each thumbs[p.name].items as t (t.id)}
          <div class="relative">
            {#if t.thumb}
              <img
                src={t.thumb}
                alt="注册预览 #{t.id}"
                class="w-14 h-14 rounded-md border border-hairline-dark object-cover"
              />
            {:else}
              <div
                class="w-14 h-14 rounded-md border border-dashed border-hairline-dark
                  flex items-center justify-center text-[10px] text-muted"
                title="旧数据，注册时未保存预览图"
              >
                无预览
              </div>
            {/if}
            <button
              type="button"
              class="absolute top-0 right-0 w-5 h-5 rounded-full bg-black/60 text-white
                text-xs leading-none cursor-pointer flex items-center justify-center
                hover:bg-trading-down"
              title="删除该条向量"
              onclick={() => delVector(p, t.id)}
            >×</button>
          </div>
        {/each}
      {/if}
    </div>
  {/if}
{/snippet}

{#snippet actions(p)}
  <div class="flex flex-wrap gap-1.5">
    <button class="btn-chip" onclick={() => toggleThumbs(p.name)}>
      {expanded === p.name ? '收起预览' : '预览'}
    </button>
    <button class="btn-chip" onclick={() => openAliasInput(p.name)}>添加别名</button>
    <button class="btn-chip" onclick={() => merge(p)}>合并</button>
    <button class="btn-danger" onclick={() => del(p.name)}>删除</button>
  </div>
{/snippet}

<div class="card">
  {#if needToken}
    <div class="msg msg-err">此页面需要管理员令牌（服务端已设置 ADMIN_TOKEN）</div>
    <AdminToken onsave={() => refresh()} />
  {/if}
  <div class="flex flex-wrap items-end gap-2">
    <div class="flex-1 min-w-40">
      <label class="field-label" for="search">搜索</label>
      <input
        id="search"
        type="text"
        class="field-input"
        bind:value={query}
        oninput={onsearch}
        placeholder="按名字或别名过滤"
      />
    </div>
    <button class="btn-secondary h-11 sm:h-10" onclick={runIssues} disabled={issuesLoading}>
      {issuesLoading ? '检查中…' : '库体检'}
    </button>
    <button class="btn-secondary h-11 sm:h-10" onclick={doExport}>导出</button>
    <button
      class="btn-secondary h-11 sm:h-10"
      onclick={() => importInput.click()}
      disabled={importing}
    >
      {importing ? '导入中…' : '导入'}
    </button>
    <input
      bind:this={importInput}
      type="file"
      accept=".zip"
      class="hidden"
      onchange={onImportFile}
    />
  </div>
  {#if issues}
    <div class="rounded-lg border border-hairline-dark p-4 flex flex-col gap-3">
      <div class="flex items-center gap-2">
        <span class="text-sm font-medium text-white">检查结果</span>
        <button
          class="btn-chip ml-auto"
          onclick={() => {
            issues = null
          }}
        >收起</button>
      </div>
      {#if issues.cross_name.length === 0 && issues.outliers.length === 0}
        <div class="text-sm text-trading-up">未发现疑似重复或离群向量</div>
      {:else}
        {#if issues.cross_name.length}
          <div>
            <div class="text-xs text-muted mb-1.5">
              跨名高相似（疑似同一人，建议合并）· {issues.cross_name.length} 组
            </div>
            <div class="flex flex-col gap-1.5">
              {#each issues.cross_name as pair (pair.a + '-' + pair.b)}
                <div class="flex flex-wrap items-center gap-2 text-sm">
                  <span class="font-num text-xs text-muted font-num">#{pair.a}</span>
                  <span class="text-white">{pair.a_name}</span>
                  <span class="font-num text-xs text-primary">≈ {pair.sim.toFixed(4)}</span>
                  <span class="text-white">{pair.b_name}</span>
                  <span class="font-num text-xs text-muted">#{pair.b}</span>
                  <button class="btn-chip" onclick={() => mergePair(pair, pair.a_name)}>
                    合并到 {pair.a_name}
                  </button>
                  <button class="btn-chip" onclick={() => mergePair(pair, pair.b_name)}>
                    合并到 {pair.b_name}
                  </button>
                </div>
              {/each}
            </div>
          </div>
        {/if}
        {#if issues.outliers.length}
          <div>
            <div class="text-xs text-muted mb-1.5">
              同名离群向量（疑似误注册，建议核对后删除）· {issues.outliers.length} 条
            </div>
            <div class="flex flex-col gap-1.5">
              {#each issues.outliers as o (o.id)}
                <div class="flex flex-wrap items-center gap-2 text-sm">
                  <span class="text-white">{o.name}</span>
                  <span class="font-num text-xs text-muted">#{o.id}</span>
                  <span class="font-num text-xs text-trading-down">
                    均值相似度 {o.sim_to_mean.toFixed(4)}
                  </span>
                  <button class="btn-danger" onclick={() => delOutlier(o)}>删除</button>
                </div>
              {/each}
            </div>
          </div>
        {/if}
      {/if}
    </div>
  {/if}
  {#if msg}
    <div class="msg {msgOk ? 'msg-ok' : 'msg-err'}">{msg}</div>
  {/if}
  {#if loading}
    <div class="text-sm text-muted py-6 text-center">加载中…</div>
  {:else if persons.length === 0}
    <div class="text-sm text-muted py-6 text-center">
      人脸库为空{currentQ ? '，无匹配结果' : ''}
    </div>
  {:else}
    <div class="md:hidden flex flex-col gap-3">
      {#each persons as p (p.name)}
        <div class="rounded-lg border border-hairline-dark p-4 flex flex-col gap-3">
          <div class="flex items-center justify-between gap-3">
            <span class="text-white font-medium min-w-0 truncate">{p.name}</span>
            <span class="shrink-0 text-xs text-muted">{p.count} 向量</span>
          </div>
          {@render aliasChips(p)}
          {@render aliasEditor(p)}
          {@render thumbGrid(p)}
          {@render actions(p)}
        </div>
      {/each}
    </div>
    <div class="hidden md:block rounded-lg border border-hairline-dark">
      <table class="w-full text-sm">
        <thead>
          <tr class="text-xs font-medium text-muted border-b border-hairline-dark">
            <th class="text-left px-4 py-3 font-medium">名字</th>
            <th class="text-right px-4 py-3 font-medium">向量数</th>
            <th class="px-4 py-3 w-40"></th>
          </tr>
        </thead>
        <tbody class="divide-y divide-hairline-dark">
          {#each persons as p (p.name)}
            <tr class="hover:bg-surface-elevated/50 align-top">
              <td class="px-4 py-3">
                <div class="flex flex-col gap-1.5">
                  <span class="text-white font-medium">{p.name}</span>
                  {@render aliasChips(p)}
                  {@render aliasEditor(p)}
                  {@render thumbGrid(p)}
                </div>
              </td>
              <td class="px-4 py-3 text-right font-num text-body">{p.count}</td>
              <td class="px-4 py-3 text-right whitespace-nowrap">
                {@render actions(p)}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
    <div class="flex items-center justify-between gap-3 text-xs text-muted">
      <span>共 {total} 人</span>
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
