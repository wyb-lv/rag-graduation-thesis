import type { AccountItem, CreateAccountInput, UpdateAccountInput, UserRoleType } from '@/types'
import { authFetch } from '@/services/authService'

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:9090').replace(/\/+$/, '')
const STORAGE_KEY = 'mock_accounts_db'

const INITIAL_MOCK_ACCOUNTS: AccountItem[] = [
  {
    id: 1,
    username: 'admin',
    email: 'admin@hoasen.edu.vn',
    departmentId: 1,
    departmentName: 'Ban Giám Hiệu',
    role: 'ADMIN',
    isActive: true,
    createdAt: '2026-10-15 08:30',
  },
  {
    id: 2,
    username: 'sinhvien_cntt',
    email: 'sv.cntt@hoasen.edu.vn',
    departmentId: 2,
    departmentName: 'Khoa Công Nghệ Thông Tin',
    role: 'USER',
    isActive: true,
    createdAt: '2026-10-18 10:15',
  },
  {
    id: 3,
    username: 'sinhvien_01',
    email: 'sv01@hoasen.edu.vn',
    departmentId: 2,
    departmentName: 'Khoa Công Nghệ Thông Tin',
    role: 'USER',
    isActive: false, // Soft deleted account example
    createdAt: '2026-10-20 14:00',
  },
]

function normalizeRole(role: unknown): UserRoleType {
  return role === 'ADMIN' ? 'ADMIN' : 'USER'
}

interface RawBackendAccount {
  id: number
  username: string
  email: string
  departmentId?: number | null
  departmentName?: string
  role?: unknown
  active?: boolean
  isActive?: boolean
  createdAt?: string
  updatedAt?: string
}

function mapAccount(raw: RawBackendAccount): AccountItem {
  const departmentId = Number(raw.departmentId ?? 1)
  return {
    id: raw.id,
    username: raw.username,
    email: raw.email,
    departmentId,
    departmentName: raw.departmentName || getDeptName(departmentId),
    role: normalizeRole(raw.role),
    isActive: raw.active ?? raw.isActive ?? true,
    createdAt: raw.createdAt,
    updatedAt: raw.updatedAt,
  }
}

async function readErrorMessage(res: Response, fallback: string): Promise<string> {
  if (res.status === 401) return 'Phiên đăng nhập đã hết hạn, vui lòng đăng nhập lại.'
  if (res.status === 403) return 'Chỉ quản trị viên (ADMIN) mới có quyền quản lý tài khoản.'
  const body = await res.json().catch(() => ({}))
  const msg = body.message || body.detail || body.error
  return typeof msg === 'string' && msg.trim() ? msg : `${fallback} (${res.status})`
}

function getLocalAccounts(): AccountItem[] {
  const data = localStorage.getItem(STORAGE_KEY)
  if (!data) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(INITIAL_MOCK_ACCOUNTS))
    return INITIAL_MOCK_ACCOUNTS
  }
  try {
    const parsed = JSON.parse(data)
    if (!Array.isArray(parsed)) return INITIAL_MOCK_ACCOUNTS
    return parsed.map(mapAccount)
  } catch {
    return INITIAL_MOCK_ACCOUNTS
  }
}

function saveLocalAccounts(items: AccountItem[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(items))
}

function getDeptName(id: number): string {
  switch (id) {
    case 1:
      return 'Ban Giám Hiệu'
    case 2:
      return 'Khoa Công Nghệ Thông Tin'
    case 3:
      return 'Phòng Đào Tạo'
    case 4:
      return 'Phòng Công Tác Sinh Viên'
    default:
      return `Phòng ban #${id}`
  }
}

// 1. Lấy danh sách tài khoản
export async function fetchAccounts(): Promise<AccountItem[]> {
  try {
    const res = await authFetch(`${API_BASE_URL}/api/accounts?size=100`)
    if (res.ok) {
      const data = await res.json()
      const items = Array.isArray(data) ? data : Array.isArray(data?.content) ? data.content : null
      if (items) return items.map(mapAccount)
    }
    if (res.status < 500) {
      throw new Error(await readErrorMessage(res, 'Không tải được danh sách tài khoản'))
    }
  } catch (err) {
    if (!(err instanceof TypeError)) throw err
    // Backend chưa chạy -> dùng mock
  }
  return getLocalAccounts()
}

// 2. Tạo tài khoản mới
export async function createAccount(input: CreateAccountInput): Promise<AccountItem> {
  try {
    const res = await authFetch(`${API_BASE_URL}/api/accounts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        username: input.username,
        email: input.email,
        departmentId: input.departmentId,
        position: '',
        role: normalizeRole(input.role),
        tempPassword: input.password || '12345678',
      }),
    })
    if (res.ok) return mapAccount(await res.json())
    if (res.status < 500) {
      throw new Error(await readErrorMessage(res, 'Không tạo được tài khoản'))
    }
  } catch (err) {
    if (!(err instanceof TypeError)) throw err
    // Fallback Mock
  }

  const list = getLocalAccounts()
  if (list.some((a) => a.username.toLowerCase() === input.username.toLowerCase())) {
    throw new Error('Tên đăng nhập đã tồn tại')
  }
  const newAccount: AccountItem = {
    id: Date.now(),
    username: input.username,
    email: input.email,
    departmentId: input.departmentId,
    departmentName: getDeptName(input.departmentId),
    role: normalizeRole(input.role),
    isActive: true,
    createdAt: new Date().toLocaleString('vi-VN'),
  }
  list.unshift(newAccount)
  saveLocalAccounts(list)
  return newAccount
}

// 3. Cập nhật tài khoản
export async function updateAccount(id: number, input: UpdateAccountInput): Promise<AccountItem> {
  try {
    const res = await authFetch(`${API_BASE_URL}/api/accounts/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        email: input.email,
        departmentId: input.departmentId,
        position: '',
        role: normalizeRole(input.role),
      }),
    })
    if (!res.ok) {
      if (res.status < 500) {
        throw new Error(await readErrorMessage(res, 'Không cập nhật được tài khoản'))
      }
      throw new TypeError('Backend unavailable')
    }
    const updated = mapAccount(await res.json())
    if (input.password?.trim()) {
      const resetRes = await authFetch(`${API_BASE_URL}/api/accounts/${id}/reset-password`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tempPassword: input.password.trim() }),
      })
      if (!resetRes.ok && resetRes.status < 500) {
        throw new Error(await readErrorMessage(resetRes, 'Không đặt lại được mật khẩu'))
      }
    }
    return updated
  } catch (err) {
    if (!(err instanceof TypeError)) throw err
    // Fallback Mock
  }

  const list = getLocalAccounts()
  const idx = list.findIndex((a) => a.id === id)
  if (idx === -1) throw new Error('Không tìm thấy tài khoản')

  list[idx] = {
    ...list[idx],
    ...input,
    role: input.role ? normalizeRole(input.role) : list[idx].role,
    departmentName: input.departmentId ? getDeptName(input.departmentId) : list[idx].departmentName,
    updatedAt: new Date().toLocaleString('vi-VN'),
  }
  saveLocalAccounts(list)
  return list[idx]
}

async function setAccountStatus(id: number, active: boolean): Promise<void> {
  try {
    const res = await authFetch(`${API_BASE_URL}/api/accounts/${id}/status`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ active }),
    })
    if (res.ok) return
    if (res.status < 500) {
      throw new Error(await readErrorMessage(res, active ? 'Không khôi phục được tài khoản' : 'Không vô hiệu hoá được tài khoản'))
    }
  } catch (err) {
    if (!(err instanceof TypeError)) throw err
    // Fallback Mock
  }

  const list = getLocalAccounts()
  const idx = list.findIndex((a) => a.id === id)
  if (idx !== -1) {
    list[idx].isActive = active
    list[idx].updatedAt = new Date().toLocaleString('vi-VN')
    saveLocalAccounts(list)
  }
}

// 4. SOFT DELETE: Vô hiệu hoá tài khoản (active = false, không xóa dòng khỏi CSDL)
export async function softDeleteAccount(id: number): Promise<void> {
  return setAccountStatus(id, false)
}

// 5. RESTORE: Kích hoạt lại tài khoản đã xóa mềm
export async function restoreAccount(id: number): Promise<void> {
  return setAccountStatus(id, true)
}
