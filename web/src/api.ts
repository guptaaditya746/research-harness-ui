export const api = '/api';

/** GET (or the given request) as JSON; a non-2xx status throws with the status and body. */
export const getJson = async <T,>(url: string, init?: RequestInit): Promise<T> => {
  const response = await fetch(url, init);
  if (!response.ok) throw new Error(`${response.status} ${await response.text().catch(() => '')}`.trim());
  return response.json() as Promise<T>;
};
