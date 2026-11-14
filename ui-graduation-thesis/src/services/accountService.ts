import type { AccountItem, CreateAccountInput, UpdateAccountInput } from '@/types'
import { authFetch } from '@/services/authService'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:9090'
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
    username: 'manager_cntt',
    email: 'cntt.manager@hoasen.edu.vn',
    departmentId: 2,
    departmentName: 'Khoa Công Nghệ Thông Tin',
    role: 'MANAGER',
    isActive: true,
    createdAt: '2026-10-18 10:15',
  },
  {
    id: 3,
    username: 'staff_gv01',
    email: 'gv01@hoasen.edu.vn',
    departmentId: 2,
    departmentName: 'Khoa Công Nghệ Thông Tin',
    role: 'EMPLOYEE',
    isActive: false, // Soft deleted account example
    createdAt: '2026-10-20 14:00',
  },
]

function getLocalAccounts(): AccountItem[] {
  const data = localStorage.getItem(STORAGE_KEY)
  if (!data) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(INITIAL_MOCK_ACCOUNTS))
    return INITIAL_MOCK_ACCOUNTS
  }
  try {
    return JSON.parse(data)
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
    const res = await authFetch(`${API_BASE_URL}/api/users`)
    if (res.ok) {
      const data = await res.json()
      if (Array.isArray(data)) return data
    }
  } catch {
    // Backend chưa chạy -> dùng mock
  }
  return getLocalAccounts()
}

// 2. Tạo tài khoản mới
export async function createAccount(input: CreateAccountInput): Promise<AccountItem> {
  try {
    const res = await authFetch(`${API_BASE_URL}/api/users`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(input),
    })
    if (res.ok) return await res.json()
  } catch {
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
    role: input.role,
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
    const res = await authFetch(`${API_BASE_URL}/api/users/${id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(input),
    })
    if (res.ok) return await res.json()
  } catch {
    // Fallback Mock
  }

  const list = getLocalAccounts()
  const idx = list.findIndex((a) => a.id === id)
  if (idx === -1) throw new Error('Không tìm thấy tài khoản')

  list[idx] = {
    ...list[idx],
    ...input,
    departmentName: input.departmentId ? getDeptName(input.departmentId) : list[idx].departmentName,
    updatedAt: new Date().toLocaleString('vi-VN'),
  }
  saveLocalAccounts(list)
  return list[idx]
}

// 4. SOFT DELETE: Vô hiệu hoá tài khoản (isActive = false, không xóa dòng khỏi CSDL)
export async function softDeleteAccount(id: number): Promise<void> {
  try {
    const res = await authFetch(`${API_BASE_URL}/api/users/${id}/deactivate`, {
      method: 'PATCH',
    })
    if (res.ok) return
  } catch {
    // Fallback Mock
  }

  const list = getLocalAccounts()
  const idx = list.findIndex((a) => a.id === id)
  if (idx !== -1) {
    list[idx].isActive = false
    list[idx].updatedAt = new Date().toLocaleString('vi-VN')
    saveLocalAccounts(list)
  }
}

// 5. RESTORE: Kích hoạt lại tài khoản đã xóa mềm
export async function restoreAccount(id: number): Promise<void> {
  try {
    const res = await authFetch(`${API_BASE_URL}/api/users/${id}/activate`, {
      method: 'PATCH',
    })
    if (res.ok) return
  } catch {
    // Fallback Mock
  }

  const list = getLocalAccounts()
  const idx = list.findIndex((a) => a.id === id)
  if (idx !== -1) {
    list[idx].isActive = true
    list[idx].updatedAt = new Date().toLocaleString('vi-VN')
    saveLocalAccounts(list)
  }
}
