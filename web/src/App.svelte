<script>
  import Recognize from './lib/Recognize.svelte'
  import UploadFace from './lib/UploadFace.svelte'
  import Compare from './lib/Compare.svelte'
  import Manage from './lib/Manage.svelte'
  import History from './lib/History.svelte'

  const tabs = [
    { id: 'recognize', label: '人脸识别' },
    { id: 'upload', label: '上传人脸' },
    { id: 'compare', label: '人脸比对' },
    { id: 'manage', label: '人脸库管理' },
    { id: 'history', label: '识别历史' }
  ]

  let active = $state('recognize')

  // 拖放未落在选择框上时阻止浏览器打开图片
  $effect(() => {
    const prevent = (e) => e.preventDefault()
    window.addEventListener('dragover', prevent)
    window.addEventListener('drop', prevent)
    return () => {
      window.removeEventListener('dragover', prevent)
      window.removeEventListener('drop', prevent)
    }
  })
</script>

<div class="min-h-screen flex flex-col">
  <header class="h-16 bg-canvas-dark border-b border-hairline-dark sticky top-0 z-10">
    <div
      class="max-w-5xl mx-auto h-full flex items-center gap-3
        pl-[max(1rem,env(safe-area-inset-left))] pr-[max(1rem,env(safe-area-inset-right))] md:px-6"
    >
      <div class="flex items-center gap-2 select-none">
        <svg viewBox="0 0 24 24" class="w-6 h-6 fill-primary" aria-hidden="true">
          <path d="M12 2 2 12l10 10 10-10L12 2Zm0 4.5L16.5 11 12 15.5 7.5 11 12 6.5Z" />
        </svg>
        <span class="text-primary text-lg font-bold tracking-wide">FaceFind</span>
      </div>
      <span class="hidden sm:inline text-hairline-dark">|</span>
      <span class="hidden sm:inline text-sm text-muted-strong font-medium">人脸识别</span>
      <a
        href="docs"
        class="ml-auto text-sm font-medium text-muted hover:text-primary transition-colors
          py-2 -mr-1 px-1"
      >
        API 文档
      </a>
    </div>
  </header>

  <main class="flex-1 w-full max-w-5xl mx-auto px-4 md:px-6 pt-6 pb-16 md:pt-12">
    <section class="mb-6 md:mb-8">
      <h1 class="text-2xl sm:text-3xl md:text-4xl font-bold text-white tracking-tight leading-tight">
        <span class="text-primary">人脸</span>识别引擎
      </h1>
      <p class="mt-2 text-sm text-muted">
        上传图片即可识别、注册、比对与管理本地人脸向量库
      </p>
    </section>

    <nav
      class="sticky top-16 z-[5] bg-canvas-dark flex gap-1 border-b border-hairline-dark mb-6
        overflow-x-auto no-scrollbar -mx-4 px-4 md:mx-0 md:px-0"
    >
      {#each tabs as t (t.id)}
        <button
          class="relative px-4 h-11 text-sm font-medium whitespace-nowrap cursor-pointer transition-colors
            {active === t.id ? 'text-primary' : 'text-muted hover:text-body'}"
          onclick={() => (active = t.id)}
        >
          {t.label}
          {#if active === t.id}
            <span class="absolute inset-x-2 -bottom-px h-0.5 bg-primary rounded-full"></span>
          {/if}
        </button>
      {/each}
    </nav>

    {#if active === 'recognize'}
      <Recognize />
    {:else if active === 'upload'}
      <UploadFace />
    {:else if active === 'compare'}
      <Compare />
    {:else if active === 'manage'}
      <Manage />
    {:else}
      <History />
    {/if}
  </main>

  <footer class="bg-surface-card border-t border-hairline-dark">
    <div
      class="max-w-5xl mx-auto px-4 md:px-6 pt-8 flex flex-col md:flex-row gap-2 md:items-center
        pb-[max(2rem,env(safe-area-inset-bottom))]"
    >
      <span class="text-sm font-semibold text-white">FaceFind 人脸识别</span>
      <span class="text-sm text-muted">本地人脸向量检索 · 数据不出本机</span>
      <span class="md:ml-auto text-xs text-muted">© 2026</span>
    </div>
  </footer>
</div>
