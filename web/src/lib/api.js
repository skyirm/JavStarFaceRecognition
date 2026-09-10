async function request(path, options = {}) {
  const res = await fetch(path, options)
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const data = await res.json()
      if (data && data.detail) detail = data.detail
    } catch {
      // keep status text
    }
    throw new Error(detail)
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

  compare: (file1, file2) =>
    request('api/compare', {
      method: 'POST',
      body: form(['file1', file1], ['file2', file2])
    })
}
