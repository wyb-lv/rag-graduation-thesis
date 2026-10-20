/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string
  readonly VITE_APP_TITLE?: string
  readonly VITE_USER_NAME?: string
  readonly VITE_USER_STUDENT_ID?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
