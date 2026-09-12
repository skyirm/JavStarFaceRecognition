let adminToken = localStorage.getItem('admin_token') ?? ''

export function getAdminToken() {
  return adminToken
}

export function setAdminToken(t) {
  adminToken = t
  if (t) localStorage.setItem('admin_token', t)
  else localStorage.removeItem('admin_token')
}

async function request(path, options = {}) {
  if (adminToken) {
    options.headers = { ...(options.headers ?? {}), 'X-Admin-Token': adminToken }
  }
  const res = await fetch(path, options)
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const data = await res.json()
      if (data && data.detail) detail = data.detail
    } catch {
      // keep status text
    }
    const err = new Error(detail)
    err.status = res.status
    throw err
  }
  return res.json()
}

function form(...pairs) {
  const fd = new FormData()
  for (const [k, v] of pairs) fd.append(k, v)
  return fd
}

export const api = {
  recognize: (file) =>
    request('api/recognize', { method: 'POST', body: form(['file', file]) }),

  uploadFace: (name, file) =>
    request(`api/faces?name=${encodeURIComponent(name)}`, {
      method: 'POST',
      body: form(['file', file])
    }),

  listFaces: () => request('api/faces'),

  suggest: (q) => request(`api/faces/suggest?q=${encodeURIComponent(q)}`),

  deleteFace: (name) =>
    request(`api/faces/${encodeURIComponent(name)}`, { method: 'DELETE' }),

  addAlias: (name, alias) =>
    request(`api/faces/${encodeURIComponent(name)}/aliases`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ alias })
    }),

  removeAlias: (name, alias) =>
    request(
      `api/faces/${encodeURIComponent(name)}/aliases/${encodeURIComponent(alias)}`,
      { method: 'DELETE' }
    ),

  mergeFace: (source, target) =>
    request('api/faces/merge', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source, target })
    }),

  compare: (file1, file2) =>
    request('api/compare', {
      method: 'POST',
      body: form(['file1', file1], ['file2', file2])
    })
}
