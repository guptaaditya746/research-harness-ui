import { readdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';

const assets = join(process.cwd(), 'dist', 'assets');
const files = await readdir(assets);
const scripts = files.filter(file => file.endsWith('.js'));
if (!scripts.some(file => file.startsWith('PublicApp-'))) {
  throw new Error('PublicApp bundle missing; build with VITE_PUBLIC_INTERACTIVE=true');
}
const bundle = (await Promise.all(scripts.map(file => readFile(join(assets, file), 'utf8')))).join('\n');
for (const forbidden of ['/benchmark', '/corpus/papers', '/settings', '/promote', '/rating', 'Gold candidates']) {
  if (bundle.includes(forbidden)) throw new Error(`Private UI control or endpoint found in public bundle: ${forbidden}`);
}
console.log('Public build contains the public entry point and no checked private controls/endpoints.');
