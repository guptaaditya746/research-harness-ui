import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './broadsheet.css';
import './styles.css';

async function start() {
  const faces = [
    new FontFace('Harness Source Serif 4', 'url(/fonts/source-serif-4-latin.woff2)', { style: 'normal', weight: '400' }),
    new FontFace('Harness Source Serif 4', 'url(/fonts/source-serif-4-latin.woff2)', { style: 'normal', weight: '600' }),
    new FontFace('Harness Source Serif 4', 'url(/fonts/source-serif-4-italic-latin.woff2)', { style: 'italic', weight: '400' }),
  ];
  await Promise.allSettled(faces.map(async face => document.fonts.add(await face.load())));
  createRoot(document.getElementById('root')!).render(<React.StrictMode><App /></React.StrictMode>);
}

void start();
