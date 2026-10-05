import React from 'react';
import { createRoot } from 'react-dom/client';
import './broadsheet.css';
import './styles.css';

async function start() {
  const element = document.getElementById('root')!;
  const { default: App } = await import('./App');
  createRoot(element).render(<React.StrictMode><App /></React.StrictMode>);
}
void start();
