<script>
  import { getAdminToken, setAdminToken } from './api.js'

  let { onsave } = $props()

  let token = $state(getAdminToken())

  function save() {
    setAdminToken(token.trim())
    onsave?.()
  }
</script>

<div class="flex gap-2 items-end">
  <div class="flex-1 min-w-0">
    <label class="field-label" for="admin-token">管理员令牌</label>
    <input
      id="admin-token"
      type="password"
      class="field-input"
      bind:value={token}
      placeholder="服务端 ADMIN_TOKEN"
      onkeydown={(e) => e.key === 'Enter' && save()}
    />
  </div>
  <button class="btn-primary" onclick={save}>保存</button>
</div>
