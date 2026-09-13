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
  const res = await requestRaw(path, options)
  return res.json()
}

async function requestRaw(path, options = {}) {
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
  return res
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

  listFaces: ({ q = '', page = 1, page_size = 50 } = {}) =>
    request(
      `api/faces?q=${encodeURIComponent(q)}&page=${page}&page_size=${page_size}`
    ),

  faceThumbs: (name) =>
    request(`api/faces/${encodeURIComponent(name)}/thumbs`),

  suggest: (q) => request(`api/faces/suggest?q=${encodeURIComponent(q)}`),

  deleteFace: (name) =>
    request(`api/faces/${encodeURIComponent(name)}`, { method: 'DELETE' }),

  deleteVector: (name, id) =>
    request(`api/faces/${encodeURIComponent(name)}/vectors/${id}`, {
      method: 'DELETE'
    }),

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

  faceIssues: ({ merge_threshold, outlier_threshold } = {}) => {
    const ps = new URLSearchParams()
    if (merge_threshold != null) ps.set('merge_threshold', merge_threshold)
    if (outlier_threshold != null) ps.set('outlier_threshold', outlier_threshold)
    const qs = ps.toString()
    return request(`api/faces/issues${qs ? `?${qs}` : ''}`)
  },

  history: ({ q = '', unknown = false, page = 1, page_size = 50 } = {}) =>
    request(
      `api/history?q=${encodeURIComponent(q)}&unknown=${unknown}&page=${page}&page_size=${page_size}`
    ),

  deleteHistory: (id) =>
    request(`api/history/${id}`, { method: 'DELETE' }),

  clearHistory: () => request('api/history', { method: 'DELETE' }),

  exportLibrary: async () => {
    const res = await requestRaw('api/library/export')
    return res.blob()
  },

  importLibrary: (file) =>
    request('api/library/import', { method: 'POST', body: form(['file', file]) }),

  compare: (file1, file2) =>
    request('api/compare', {
      method: 'POST',
      body: form(['file1', file1], ['file2', file2])
    })
}
