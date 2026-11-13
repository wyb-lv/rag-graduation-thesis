import { useState, useEffect, useMemo, type FormEvent } from 'react'
import type { AccountItem, UserRoleType, CreateAccountInput, UpdateAccountInput } from '@/types'
import {
  fetchAccounts,
  createAccount,
  updateAccount,
  softDeleteAccount,
  restoreAccount,
} from '@/services/accountService'

const DEPARTMENTS = [
  { id: 1, name: 'Ban Giám Hiệu' },
  { id: 2, name: 'Khoa Công Nghệ Thông Tin' },
  { id: 3, name: 'Phòng Đào Tạo' },
  { id: 4, name: 'Phòng Công Tác Sinh Viên' },
]

export function AccountManagementView() {
  const [accounts, setAccounts] = useState<AccountItem[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [successMsg, setSuccessMsg] = useState<string | null>(null)

  // Search & Filter
  const [searchTerm, setSearchTerm] = useState('')
  const [roleFilter, setRoleFilter] = useState<'ALL' | UserRoleType>('ALL')
  const [statusFilter, setStatusFilter] = useState<'ALL' | 'ACTIVE' | 'INACTIVE'>('ALL')

  // Create / Edit modal state
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [editingAccount, setEditingAccount] = useState<AccountItem | null>(null)
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    departmentId: 1,
    role: 'EMPLOYEE' as UserRoleType,
  })

  // Soft Delete confirmation modal state
  const [confirmDeleteAcc, setConfirmDeleteAcc] = useState<AccountItem | null>(null)

  const showSuccess = (msg: string) => {
    setSuccessMsg(msg)
    setTimeout(() => setSuccessMsg(null), 3000)
  }

  const loadData = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await fetchAccounts()
      setAccounts(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Không tải được danh sách tài khoản')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
  }, [])

  // Filtered accounts based on search and filters
  const filteredAccounts = useMemo(() => {
    return accounts.filter((acc) => {
      const matchSearch =
        acc.username.toLowerCase().includes(searchTerm.toLowerCase()) ||
        acc.email.toLowerCase().includes(searchTerm.toLowerCase())
      const matchRole = roleFilter === 'ALL' || acc.role === roleFilter
      const matchStatus =
        statusFilter === 'ALL' ||
        (statusFilter === 'ACTIVE' && acc.isActive) ||
        (statusFilter === 'INACTIVE' && !acc.isActive)

      return matchSearch && matchRole && matchStatus
    })
  }, [accounts, searchTerm, roleFilter, statusFilter])

  const handleOpenCreate = () => {
    setEditingAccount(null)
    setFormData({
      username: '',
      email: '',
      password: '',
      departmentId: 1,
      role: 'EMPLOYEE',
    })
    setIsModalOpen(true)
  }

  const handleOpenEdit = (acc: AccountItem) => {
    setEditingAccount(acc)
    setFormData({
      username: acc.username,
      email: acc.email,
      password: '',
      departmentId: acc.departmentId || 1,
      role: acc.role,
    })
    setIsModalOpen(true)
  }

  const handleSubmitForm = async (e: FormEvent) => {
    e.preventDefault()
    setError(null)
    try {
      if (editingAccount) {
        const updatePayload: UpdateAccountInput = {
          email: formData.email,
          departmentId: formData.departmentId,
          role: formData.role,
        }
        if (formData.password.trim()) {
          updatePayload.password = formData.password
        }
        await updateAccount(editingAccount.id, updatePayload)
        showSuccess(`Đã cập nhật tài khoản ${editingAccount.username}`)
      } else {
        const createPayload: CreateAccountInput = {
          username: formData.username.trim(),
          email: formData.email.trim(),
          password: formData.password || '123456',
          departmentId: formData.departmentId,
          role: formData.role,
        }
        await createAccount(createPayload)
        showSuccess(`Đã tạo tài khoản ${createPayload.username}`)
      }
      setIsModalOpen(false)
      await loadData()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Thao tác không thành công')
    }
  }

  const handleConfirmSoftDelete = async () => {
    if (!confirmDeleteAcc) return
    try {
      await softDeleteAccount(confirmDeleteAcc.id)
      showSuccess(`Đã vô hiệu hoá tài khoản ${confirmDeleteAcc.username}`)
      setConfirmDeleteAcc(null)
      await loadData()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Xoá mềm thất bại')
    }
  }

  const handleRestore = async (acc: AccountItem) => {
    try {
      await restoreAccount(acc.id)
      showSuccess(`Đã khôi phục tài khoản ${acc.username}`)
      await loadData()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Khôi phục thất bại')
    }
  }

  return (
    <div className="flex-1 flex flex-col h-full bg-[#0e0e0e] text-white overflow-hidden">
      {/* Top Header */}
      <div className="h-14 px-6 border-b border-white/10 flex items-center justify-between bg-black flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-[#55b9ff]/10 text-[#55b9ff] flex items-center justify-center border border-[#55b9ff]/20 font-bold text-sm">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" />
            </svg>
          </div>
          <div>
            <h1 className="text-sm font-semibold text-white tracking-wide">
              Quản lý Tài khoản (Account Management)
            </h1>
            <p className="text-[11px] text-zinc-400">
              Quản lý danh sách người dùng, phân quyền và trạng thái hoạt động (Soft Delete)
            </p>
          </div>
        </div>

        <button
          type="button"
          onClick={handleOpenCreate}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-[#1e3c66] hover:bg-[#274d82] text-white border border-white/10 transition-colors shadow-sm cursor-pointer"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          Thêm tài khoản
        </button>
      </div>

      {/* Thông báo Alert */}
      {error && (
        <div className="mx-6 mt-3 px-3 py-2 bg-red-500/10 border border-red-500/20 text-red-300 text-xs rounded-lg flex items-center justify-between">
          <span>{error}</span>
          <button type="button" onClick={() => setError(null)} className="text-red-400 hover:text-white">✕</button>
        </div>
      )}
      {successMsg && (
        <div className="mx-6 mt-3 px-3 py-2 bg-emerald-500/10 border border-emerald-500/20 text-emerald-300 text-xs rounded-lg flex items-center justify-between">
          <span>{successMsg}</span>
          <button type="button" onClick={() => setSuccessMsg(null)} className="text-emerald-400 hover:text-white">✕</button>
        </div>
      )}

      {/* Bộ lọc & Tìm kiếm */}
      <div className="p-6 pb-3 flex flex-wrap items-center justify-between gap-3 flex-shrink-0">
        <div className="flex flex-wrap items-center gap-3 flex-1 min-w-[280px]">
          {/* Ô tìm kiếm */}
          <div className="relative min-w-[240px] flex-1 max-w-md">
            <svg className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-zinc-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input
              type="text"
              placeholder="Tìm theo username hoặc email..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full bg-[#181818] border border-white/10 rounded-lg pl-9 pr-3 py-1.5 text-xs text-white placeholder-zinc-500 focus:outline-none focus:border-[#55b9ff]/60"
            />
          </div>

          {/* Lọc Role */}
          <select
            value={roleFilter}
            onChange={(e) => setRoleFilter(e.target.value as 'ALL' | UserRoleType)}
            className="bg-[#181818] border border-white/10 rounded-lg px-2.5 py-1.5 text-xs text-zinc-300 focus:outline-none focus:border-[#55b9ff]/60"
          >
            <option value="ALL">Tất cả vai trò</option>
            <option value="ADMIN">ADMIN</option>
            <option value="MANAGER">MANAGER</option>
            <option value="EMPLOYEE">EMPLOYEE</option>
          </select>

          {/* Lọc Trạng thái (Active / Soft Deleted) */}
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value as 'ALL' | 'ACTIVE' | 'INACTIVE')}
            className="bg-[#181818] border border-white/10 rounded-lg px-2.5 py-1.5 text-xs text-zinc-300 focus:outline-none focus:border-[#55b9ff]/60"
          >
            <option value="ALL">Tất cả trạng thái</option>
            <option value="ACTIVE">Đang hoạt động</option>
            <option value="INACTIVE">Đã vô hiệu hoá (Soft Deleted)</option>
          </select>
        </div>

        <button
          type="button"
          onClick={loadData}
          disabled={loading}
          className="text-xs text-zinc-400 hover:text-white px-2 py-1 rounded hover:bg-white/5 transition-colors flex items-center gap-1 cursor-pointer"
        >
          <svg className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Làm mới
        </button>
      </div>

      {/* Bảng Danh sách Tài khoản */}
      <div className="flex-1 px-6 pb-6 overflow-y-auto">
        <div className="w-full border border-white/10 rounded-xl bg-[#141414] overflow-hidden">
          <table className="w-full text-left text-xs">
            <thead className="bg-[#1c1c1c] text-zinc-400 border-b border-white/5 uppercase text-[11px] font-medium tracking-wider">
              <tr>
                <th className="py-3 px-4">Tài khoản</th>
                <th className="py-3 px-4">Email</th>
                <th className="py-3 px-4">Phòng ban</th>
                <th className="py-3 px-4">Vai trò</th>
                <th className="py-3 px-4">Trạng thái</th>
                <th className="py-3 px-4">Ngày tạo</th>
                <th className="py-3 px-4 text-right">Thao tác</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 text-zinc-300">
              {loading ? (
                <tr>
                  <td colSpan={7} className="py-8 text-center text-zinc-500">
                    Đang tải danh sách tài khoản...
                  </td>
                </tr>
              ) : filteredAccounts.length === 0 ? (
                <tr>
                  <td colSpan={7} className="py-8 text-center text-zinc-500">
                    Không tìm thấy tài khoản phù hợp
                  </td>
                </tr>
              ) : (
                filteredAccounts.map((acc) => (
                  <tr key={acc.id} className={`hover:bg-white/[0.02] transition-colors ${!acc.isActive ? 'opacity-60 bg-red-950/10' : ''}`}>
                    <td className="py-3 px-4 font-medium text-white flex items-center gap-2">
                      <div className="w-7 h-7 rounded-full bg-zinc-800 text-zinc-300 border border-white/10 flex items-center justify-center font-bold text-xs">
                        {acc.username.slice(0, 2).toUpperCase()}
                      </div>
                      <span>{acc.username}</span>
                    </td>
                    <td className="py-3 px-4 text-zinc-400">{acc.email}</td>
                    <td className="py-3 px-4 text-zinc-300">{acc.departmentName || `Phòng ban #${acc.departmentId}`}</td>
                    <td className="py-3 px-4">
                      <span className={`inline-block px-2 py-0.5 rounded text-[11px] font-medium border ${
                        acc.role === 'ADMIN'
                          ? 'bg-purple-500/10 text-purple-400 border-purple-500/20'
                          : acc.role === 'MANAGER'
                          ? 'bg-blue-500/10 text-blue-400 border-blue-500/20'
                          : 'bg-zinc-800 text-zinc-300 border-white/10'
                      }`}>
                        {acc.role}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      {acc.isActive ? (
                        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                          Hoạt động
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-medium bg-red-500/10 text-red-400 border border-red-500/20">
                          <span className="w-1.5 h-1.5 rounded-full bg-red-400" />
                          Đã xoá mềm
                        </span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-zinc-500 text-[11px]">{acc.createdAt || '—'}</td>
                    <td className="py-3 px-4 text-right">
                      <div className="inline-flex items-center gap-1">
                        {/* Nút Sửa */}
                        <button
                          type="button"
                          onClick={() => handleOpenEdit(acc)}
                          title="Sửa tài khoản"
                          className="p-1.5 hover:text-white text-zinc-400 hover:bg-white/10 rounded transition-colors cursor-pointer"
                        >
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                          </svg>
                        </button>

                        {/* Nút Soft Delete */}
                        {acc.isActive ? (
                          <button
                            type="button"
                            onClick={() => setConfirmDeleteAcc(acc)}
                            title="Vô hiệu hoá (Soft Delete)"
                            className="p-1.5 hover:text-red-400 text-zinc-400 hover:bg-red-500/10 rounded transition-colors cursor-pointer"
                          >
                            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                            </svg>
                          </button>
                        ) : (
                          /* Nút Restore */
                          <button
                            type="button"
                            onClick={() => handleRestore(acc)}
                            title="Khôi phục tài khoản"
                            className="p-1.5 hover:text-emerald-400 text-emerald-500 hover:bg-emerald-500/10 rounded transition-colors cursor-pointer"
                          >
                            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                            </svg>
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* MODAL THÊM / SỬA TÀI KHOẢN */}
      {isModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-xs p-4">
          <div className="w-full max-w-md bg-[#161616] border border-white/10 rounded-2xl p-6 shadow-2xl">
            <div className="flex items-center justify-between pb-4 border-b border-white/10">
              <h2 className="text-sm font-semibold text-white">
                {editingAccount ? `Chỉnh sửa: ${editingAccount.username}` : 'Thêm tài khoản mới'}
              </h2>
              <button
                type="button"
                onClick={() => setIsModalOpen(false)}
                className="text-zinc-400 hover:text-white cursor-pointer"
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleSubmitForm} className="mt-4 space-y-3.5 text-xs">
              <div>
                <label className="block text-zinc-300 mb-1 font-medium">Tên đăng nhập (Username)</label>
                <input
                  type="text"
                  required
                  disabled={!!editingAccount}
                  placeholder="Ví dụ: nguyenvana"
                  value={formData.username}
                  onChange={(e) => setFormData({ ...formData, username: e.target.value })}
                  className="w-full bg-[#202020] border border-white/10 rounded-lg px-3 py-2 text-white placeholder-zinc-500 focus:outline-none focus:border-[#55b9ff]/60 disabled:opacity-50"
                />
              </div>

              <div>
                <label className="block text-zinc-300 mb-1 font-medium">Email trường / công vụ</label>
                <input
                  type="email"
                  required
                  placeholder="a.nguyen@hoasen.edu.vn"
                  value={formData.email}
                  onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                  className="w-full bg-[#202020] border border-white/10 rounded-lg px-3 py-2 text-white placeholder-zinc-500 focus:outline-none focus:border-[#55b9ff]/60"
                />
              </div>

              <div>
                <label className="block text-zinc-300 mb-1 font-medium">
                  {editingAccount ? 'Mật khẩu mới (Để trống nếu không đổi)' : 'Mật khẩu'}
                </label>
                <input
                  type="password"
                  placeholder={editingAccount ? '••••••••' : 'Nhập mật khẩu ban đầu'}
                  value={formData.password}
                  onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                  className="w-full bg-[#202020] border border-white/10 rounded-lg px-3 py-2 text-white placeholder-zinc-500 focus:outline-none focus:border-[#55b9ff]/60"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-zinc-300 mb-1 font-medium">Phòng ban / Khoa</label>
                  <select
                    value={formData.departmentId}
                    onChange={(e) => setFormData({ ...formData, departmentId: Number(e.target.value) })}
                    className="w-full bg-[#202020] border border-white/10 rounded-lg px-2.5 py-2 text-white focus:outline-none focus:border-[#55b9ff]/60"
                  >
                    {DEPARTMENTS.map((dept) => (
                      <option key={dept.id} value={dept.id}>{dept.name}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-zinc-300 mb-1 font-medium">Vai trò (Role)</label>
                  <select
                    value={formData.role}
                    onChange={(e) => setFormData({ ...formData, role: e.target.value as UserRoleType })}
                    className="w-full bg-[#202020] border border-white/10 rounded-lg px-2.5 py-2 text-white focus:outline-none focus:border-[#55b9ff]/60"
                  >
                    <option value="EMPLOYEE">EMPLOYEE</option>
                    <option value="MANAGER">MANAGER</option>
                    <option value="ADMIN">ADMIN</option>
                  </select>
                </div>
              </div>

              <div className="flex items-center justify-end gap-2 pt-4 border-t border-white/10">
                <button
                  type="button"
                  onClick={() => setIsModalOpen(false)}
                  className="px-3.5 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-300 transition-colors cursor-pointer"
                >
                  Hủy
                </button>
                <button
                  type="submit"
                  className="px-4 py-1.5 rounded-lg bg-[#1e3c66] hover:bg-[#274d82] text-white font-medium transition-colors cursor-pointer"
                >
                  {editingAccount ? 'Lưu thay đổi' : 'Tạo tài khoản'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* MODAL XÁC NHẬN SOFT DELETE */}
      {confirmDeleteAcc && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-xs p-4">
          <div className="w-full max-w-sm bg-[#161616] border border-red-500/30 rounded-2xl p-6 shadow-2xl">
            <div className="w-10 h-10 rounded-full bg-red-500/10 text-red-400 border border-red-500/20 flex items-center justify-center mb-3">
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </div>
            <h3 className="text-sm font-semibold text-white">Xác nhận vô hiệu hoá tài khoản?</h3>
            <p className="mt-2 text-xs text-zinc-400 leading-relaxed">
              Tài khoản <strong className="text-white">"{confirmDeleteAcc.username}"</strong> sẽ chuyển sang trạng thái <em>Đã xoá mềm (Inactive)</em> và không thể đăng nhập. Dữ liệu vẫn được giữ nguyên và có thể khôi phục lại bất kỳ lúc nào.
            </p>

            <div className="flex items-center justify-end gap-2 mt-5">
              <button
                type="button"
                onClick={() => setConfirmDeleteAcc(null)}
                className="px-3.5 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-xs transition-colors cursor-pointer"
              >
                Hủy
              </button>
              <button
                type="button"
                onClick={handleConfirmSoftDelete}
                className="px-3.5 py-1.5 rounded-lg bg-red-600 hover:bg-red-700 text-white font-medium text-xs transition-colors cursor-pointer"
              >
                Xác nhận vô hiệu hoá
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
