import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  use: { ...devices['Desktop Chrome'], baseURL: 'http://127.0.0.1:5173', viewport: { width: 1280, height: 900 } },
  webServer: [
    { command: 'uv run --python 3.12 uvicorn main:app --app-dir api --host 127.0.0.1 --port 8000', cwd: '..', port: 8000, reuseExistingServer: true, timeout: 120000 },
    { command: 'corepack pnpm dev', port: 5173, reuseExistingServer: true, timeout: 120000 },
  ],
});
