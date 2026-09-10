<script>
  let { label = '选择图片', file = $bindable(null) } = $props()

  let url = $state('')

  function onpick(e) {
    const f = e.target.files[0] ?? null
    if (url) URL.revokeObjectURL(url)
    file = f
    url = f ? URL.createObjectURL(f) : ''
  }
</script>

<div>
  <label class="field-label">
    {label}
    <input type="file" accept="image/jpeg,image/png,image/webp" onchange={onpick} />
  </label>
  {#if url}
    <img {url} alt="预览" style="margin-top: 8px" />
  {/if}
</div>
