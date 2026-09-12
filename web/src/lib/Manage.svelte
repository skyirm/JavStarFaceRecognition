<script>
  import { api } from './api.js'
  import AdminToken from './AdminToken.svelte'

  let persons = $state([])
  let query = $state('')
  let loading = $state(false)
  let msg = $state('')
  let msgOk = $state(true)
  let aliasTarget = $state(null)
  let aliasInput = $state('')
  let needToken = $state(false)

  let filtered = $derived(
    persons.filter((p) => {
      const q = query.trim().toLowerCase()
      if (!q) return true
      return (
        p.name.toLowerCase().includes(q) ||
        p.aliases.some((a) => a.toLowerCase().includes(q))
      )
    })
  )

  $effect(() => {
    refresh()
  })

  async function refresh() {
    loading = true
    msg = ''
    try {
      persons = await api.listFaces()
      needToken = false
    } catch (e) {
      if (e.status === 401) needToken = true
      msg = e.message
      msgOk = false
    }
    loading = false
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

{#snippet actions(p)}
  <div class="flex flex-wrap gap-1.5">
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
  <div>
    <label class="field-label" for="search">搜索</label>
    <input
      id="search"
      type="text"
      class="field-input"
      bind:value={query}
      placeholder="按名字或别名过滤"
    />
  </div>
  {#if msg}
    <div class="msg {msgOk ? 'msg-ok' : 'msg-err'}">{msg}</div>
  {/if}
  {#if loading}
    <div class="text-sm text-muted py-6 text-center">加载中…</div>
  {:else if filtered.length === 0}
    <div class="text-sm text-muted py-6 text-center">
      人脸库为空{query ? '，无匹配结果' : ''}
    </div>
  {:else}
    <div class="md:hidden flex flex-col gap-3">
      {#each filtered as p (p.name)}
        <div class="rounded-lg border border-hairline-dark p-4 flex flex-col gap-3">
          <div class="flex items-center justify-between gap-3">
            <span class="text-white font-medium min-w-0 truncate">{p.name}</span>
            <span class="shrink-0 text-xs text-muted">{p.count} 向量</span>
          </div>
          {@render aliasChips(p)}
          {@render aliasEditor(p)}
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
          {#each filtered as p (p.name)}
            <tr class="hover:bg-surface-elevated/50 align-top">
              <td class="px-4 py-3">
                <div class="flex flex-col gap-1.5">
                  <span class="text-white font-medium">{p.name}</span>
                  {@render aliasChips(p)}
                  {@render aliasEditor(p)}
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
    <div class="text-xs text-muted">共 {filtered.length} 人</div>
  {/if}
</div>
