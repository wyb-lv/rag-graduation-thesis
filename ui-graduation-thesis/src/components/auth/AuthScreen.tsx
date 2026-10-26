import { useState, type FormEvent } from 'react'
import { useChat } from '@/context/ChatContext'
import { loginApi, registerApi } from '@/services/authService'

export function AuthScreen() {
  const { loginSuccess } = useChat()
  const [tab, setTab] = useState<'login' | 'register'>('login')

  // Form states
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [email, setEmail] = useState('')
  const [khoa, setKhoa] = useState('')
  const [role, setRole] = useState<'EMPLOYEE' | 'MANAGER' | 'ADMIN'>('EMPLOYEE')

  const [loading, setLoading] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setErrorMessage(null)

    if (!username.trim() || !password.trim()) {
      setErrorMessage('Vui lòng điền đầy đủ thông tin.')
      return
    }

    if (tab === 'register' && !email.trim()) {
      setErrorMessage('Vui lòng nhập địa chỉ email hợp lệ.')
      return
    }

    setLoading(true)
    try {
      if (tab === 'login') {
        const res = await loginApi(username.trim(), password)
        if (khoa) localStorage.setItem('khoa', khoa)
        else localStorage.removeItem('khoa')
        loginSuccess(res.accessToken, res.username)
      } else {
        const res = await registerApi(username.trim(), email.trim(), password, 1, role)
        loginSuccess(res.accessToken, res.username)
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Đã có lỗi xảy ra khi xác thực'
      setErrorMessage(msg)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen w-full bg-black text-white flex flex-col justify-center items-center p-4 selection:bg-[#55b9ff]/30">
      {/* Background radial glow */}
      <div className="fixed inset-0 pointer-events-none bg-[radial-gradient(circle_at_50%_20%,rgba(85,185,255,0.08),transparent_50%)]" />

      <div className="relative w-full max-w-md bg-[#181818] border border-white/10 rounded-2xl p-6 sm:p-8 shadow-2xl backdrop-blur-sm">
        {/* Brand Header */}
        <div className="text-center mb-6">
          <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-[#55b9ff]/10 text-[#55b9ff] mb-3 border border-[#55b9ff]/20">
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 11c0 3.517-1.009 6.799-2.753 9.571m-3.44-2.04l.054-.09A13.916 13.916 0 008 11a4 4 0 118 0c0 1.017-.07 2.019-.203 3m-2.118 6.844A21.88 21.88 0 0015.171 17m3.839 1.132c.645-2.266.99-4.659.99-7.132A8 8 0 004 11m0 0a8.003 8.003 0 0115.357-2m1.571 2a8.006 8.006 0 01-.84 3.55" />
            </svg>
          </div>
          <h1 className="text-xl sm:text-2xl font-semibold tracking-tight text-white">
            Trợ lý AI ĐH Hoa Sen
          </h1>
          <p className="text-xs text-[#8e8e8e] mt-1">
            Hệ thống Tra cứu Quy chế, Quét OCR & Hỏi đáp RAG
          </p>
        </div>

        {/* Tab switch */}
        <div className="flex bg-[#212121] p-1 rounded-xl mb-6 border border-white/5">
          <button
            type="button"
            onClick={() => {
              setTab('login')
              setErrorMessage(null)
            }}
            className={`flex-1 py-2 text-sm font-medium rounded-lg transition-all ${
              tab === 'login'
                ? 'bg-[#2f2f2f] text-white shadow-sm'
                : 'text-[#8e8e8e] hover:text-white'
            }`}
          >
            Đăng nhập
          </button>
          <button
            type="button"
            onClick={() => {
              setTab('register')
              setErrorMessage(null)
            }}
            className={`flex-1 py-2 text-sm font-medium rounded-lg transition-all ${
              tab === 'register'
                ? 'bg-[#2f2f2f] text-white shadow-sm'
                : 'text-[#8e8e8e] hover:text-white'
            }`}
          >
            Đăng ký mới
          </button>
        </div>

        {/* Error notification */}
        {errorMessage && (
          <div className="mb-4 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-xs flex items-center gap-2">
            <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>{errorMessage}</span>
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-medium text-[#b4b4b4] mb-1.5">
              Tên đăng nhập
            </label>
            <input
              type="text"
              required
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="VD: sinhvien_k22"
              className="w-full bg-[#212121] border border-white/10 rounded-lg px-3.5 py-2.5 text-sm text-white placeholder-[#666] focus:outline-none focus:border-[#55b9ff] focus:ring-1 focus:ring-[#55b9ff] transition-all"
            />
          </div>

          {tab === 'register' && (
            <div>
              <label className="block text-xs font-medium text-[#b4b4b4] mb-1.5">
                Email sinh viên / cán bộ
              </label>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="VD: sinhvien@hoasen.edu.vn"
                className="w-full bg-[#212121] border border-white/10 rounded-lg px-3.5 py-2.5 text-sm text-white placeholder-[#666] focus:outline-none focus:border-[#55b9ff] focus:ring-1 focus:ring-[#55b9ff] transition-all"
              />
            </div>
          )}

          <div>
            <label className="block text-xs font-medium text-[#b4b4b4] mb-1.5">
              Mật khẩu
            </label>
            <input
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              className="w-full bg-[#212121] border border-white/10 rounded-lg px-3.5 py-2.5 text-sm text-white placeholder-[#666] focus:outline-none focus:border-[#55b9ff] focus:ring-1 focus:ring-[#55b9ff] transition-all"
            />
          </div>

          {tab === 'login' && (
            <div>
              <label className="block text-xs font-medium text-[#b4b4b4] mb-1.5">
                Khóa (không bắt buộc)
              </label>
              <input
                type="number"
                min={2000}
                max={2100}
                value={khoa}
                onChange={(e) => setKhoa(e.target.value)}
                placeholder="VD: 2024 — để chỉ xem quy định áp dụng cho khóa của bạn"
                className="w-full bg-[#212121] border border-white/10 rounded-lg px-3.5 py-2.5 text-sm text-white placeholder-[#666] focus:outline-none focus:border-[#55b9ff] focus:ring-1 focus:ring-[#55b9ff] transition-all"
              />
            </div>
          )}

          {tab === 'register' && (
            <div>
              <label className="block text-xs font-medium text-[#b4b4b4] mb-1.5">
                Vai trò truy cập
              </label>
              <select
                value={role}
                onChange={(e) => setRole(e.target.value as 'EMPLOYEE' | 'MANAGER' | 'ADMIN')}
                className="w-full bg-[#212121] border border-white/10 rounded-lg px-3.5 py-2.5 text-sm text-white focus:outline-none focus:border-[#55b9ff] transition-all"
              >
                <option value="EMPLOYEE">Chuyên viên / Sinh viên (EMPLOYEE)</option>
                <option value="MANAGER">Quản lý Phòng ban (MANAGER)</option>
                <option value="ADMIN">Quản trị viên (ADMIN)</option>
              </select>
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full mt-2 py-2.5 px-4 bg-[#55b9ff] hover:bg-[#43a5e8] text-black font-semibold rounded-lg text-sm transition-colors flex items-center justify-center gap-2 disabled:opacity-60 disabled:cursor-not-allowed shadow-md shadow-[#55b9ff]/10"
          >
            {loading ? (
              <span className="inline-block w-4 h-4 border-2 border-black border-t-transparent rounded-full animate-spin" />
            ) : tab === 'login' ? (
              'Đăng nhập hệ thống'
            ) : (
              'Tạo tài khoản mới'
            )}
          </button>
        </form>
      </div>
    </div>
  )
}
