const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:9090'

export interface LoginResponse {
  accessToken: string
  refreshToken?: string
  username: string
  /** Mật khẩu tạm do admin cấp / đặt lại: phải đổi trước khi dùng hệ thống. */
  mustChangePassword: boolean
}

export async function loginApi(username: string, password: string): Promise<LoginResponse> {
  const res = await fetch(`${API_BASE_URL}/api/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })
  if (res.ok) {
    const data = await res.json()
    // Chưa đổi mật khẩu tạm thì chưa lưu phiên: AuthScreen lưu refreshToken sau khi đổi xong
    if (data.refreshToken && !data.mustChangePassword) {
      localStorage.setItem('refreshToken', data.refreshToken)
    }
    return {
      accessToken: data.accessToken || data.token,
      refreshToken: data.refreshToken,
      username: data.username || username,
      mustChangePassword: data.mustChangePassword === true,
    }
  }
  const err = await res.json().catch(() => ({}))
  throw new Error(err.message || err.detail || `Đăng nhập không thành công (${res.status})`)
}

/** Tự đổi mật khẩu bằng token vừa nhận (chưa lưu vào localStorage nên không dùng authFetch). Không thu hồi phiên đang đăng nhập. */
export async function changePasswordApi(accessToken: string, currentPassword: string, newPassword: string): Promise<void> {
  const res = await fetch(`${API_BASE_URL}/api/accounts/me/change-password`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${accessToken}` },
    body: JSON.stringify({ currentPassword, newPassword }),
  })
  if (res.ok) return
  if (res.status === 400) throw new Error('Mật khẩu mới không hợp lệ (8–72 ký tự).')
  const err = await res.json().catch(() => ({}))
  throw new Error(err.message || err.detail || `Đổi mật khẩu không thành công (${res.status})`)
}

/** Khóa của sinh viên (vd 2024), gửi kèm câu hỏi để chỉ lấy quy định áp dụng cho khóa đó. */
export function getKhoa(): number | undefined {
  const n = Number(localStorage.getItem('khoa'))
  return Number.isInteger(n) && n > 0 ? n : undefined
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
