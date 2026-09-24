import React from 'react';
import { createRoot } from 'react-dom/client';
import './broadsheet.css';
import './styles.css';

async function start() {
  const element = document.getElementById('root')!;
  if (import.meta.env.VITE_PUBLIC_INTERACTIVE === 'true') {
    const { default: PublicApp } = await import('./PublicApp');
    createRoot(element).render(<React.StrictMode><PublicApp /></React.StrictMode>);
    return;
  }
  const { default: App } = await import('./App');
  createRoot(element).render(<React.StrictMode><App /></React.StrictMode>);
}
void start();
