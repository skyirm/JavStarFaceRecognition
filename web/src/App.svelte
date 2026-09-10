<script>
  import Recognize from './lib/Recognize.svelte'
  import UploadFace from './lib/UploadFace.svelte'
  import Compare from './lib/Compare.svelte'
  import Manage from './lib/Manage.svelte'

  const tabs = [
    { id: 'recognize', label: '人脸识别' },
    { id: 'upload', label: '上传人脸' },
    { id: 'compare', label: '人脸比对' },
    { id: 'manage', label: '人脸库管理' }
  ]

  let active = $state('recognize')
</script>

<div class="container">
  <h1>人脸识别</h1>
  <nav class="tabs">
    {#each tabs as t (t.id)}
      <button class:active={active === t.id} onclick={() => (active = t.id)}>
        {t.label}
      </button>
    {/each}
  </nav>

  {#if active === 'recognize'}
    <Recognize />
  {:else if active === 'upload'}
    <UploadFace />
  {:else if active === 'compare'}
    <Compare />
  {:else}
    <Manage />
  {/if}
</div>
