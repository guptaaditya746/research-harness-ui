/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_PUBLIC_INTERACTIVE?: string;
  readonly VITE_PUBLIC_API_BASE?: string;
  readonly VITE_TURNSTILE_SITE_KEY?: string;
}
interface ImportMeta { readonly env: ImportMetaEnv }
