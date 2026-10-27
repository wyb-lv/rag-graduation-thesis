const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:9090'

export interface LoginResponse {
  accessToken: string
  refreshToken?: string
  username: string
}

export async function loginApi(username: string, password: string): Promise<LoginResponse> {
  const res = await fetch(`${API_BASE_URL}/api/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })
  if (res.ok) {
    const data = await res.json()
    if (data.refreshToken) {
      localStorage.setItem('refreshToken', data.refreshToken)
    }
    return {
      accessToken: data.accessToken || data.token,
      refreshToken: data.refreshToken,
      username: data.username || username,
    }
  }
  const err = await res.json().catch(() => ({}))
  throw new Error(err.message || err.detail || `Đăng nhập không thành công (${res.status})`)
}

export async function registerApi(
  username: string,
  email: string,
  password: string,
  departmentId = 1,
  role = 'EMPLOYEE'
): Promise<LoginResponse> {
  const res = await fetch(`${API_BASE_URL}/api/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, email, password, departmentId, role }),
  })
  if (res.ok) {
    const data = await res.json()
    if (data.refreshToken) {
      localStorage.setItem('refreshToken', data.refreshToken)
    }
    return {
      accessToken: data.accessToken || data.token,
      refreshToken: data.refreshToken,
      username: data.username || username,
    }
  }
  const err = await res.json().catch(() => ({}))
  throw new Error(err.message || err.detail || `Đăng ký không thành công (${res.status})`)
}

/** Khóa của sinh viên (vd 2024), gửi kèm câu hỏi để chỉ lấy quy định áp dụng cho khóa đó. */
/** Khóa của sinh viên (vd 2024), gửi kèm câu hỏi để chỉ lấy quy định áp dụng cho khóa đó. */
export function getKhoa(): number | undefined {
  const raw = localStorage.getItem('khoa')
  if (!raw) return undefined
  let n = Number(raw)
  if (!Number.isInteger(n)) return undefined
  // Nếu sinh viên nhập 2 chữ số (vd: Khóa 21, 22, 24) -> tự quy đổi thành 2021, 2022, 2024
  if (n >= 10 && n <= 99) {
    n += 2000
  }
  return n >= 1990 && n <= 2100 ? n : undefined
}

// Nhiều request cùng hết hạn thì chỉ refresh một lần (refresh token dùng lại sẽ bị thu hồi toàn bộ phiên)
let refreshing: Promise<boolean> | null = null

function refreshSession(): Promise<boolean> {
  refreshing ??= (async () => {
    const refreshToken = localStorage.getItem('refreshToken')
    if (!refreshToken) return false
    try {
      const res = await fetch(`${API_BASE_URL}/api/auth/refresh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refreshToken }),
      })
      if (!res.ok) return false
      const data = await res.json()
      localStorage.setItem('accessToken', data.accessToken)
      localStorage.setItem('refreshToken', data.refreshToken)
      return true
    } catch {
      return false
    }
  })().finally(() => {
    refreshing = null
  })
  return refreshing
}

/** fetch kèm Bearer token; gặp 401 thì refresh một lần rồi gọi lại, vẫn 401 thì báo phiên hết hạn (auth:expired). */
export async function authFetch(url: string, init: RequestInit = {}): Promise<Response> {
  const send = () =>
    fetch(url, {
      ...init,
      headers: { ...init.headers, Authorization: `Bearer ${localStorage.getItem('accessToken') ?? ''}` },
    })
  let res = await send()
  if (res.status === 401) {
    if (await refreshSession()) res = await send()
    if (res.status === 401) window.dispatchEvent(new Event('auth:expired'))
  }
  return res
}
