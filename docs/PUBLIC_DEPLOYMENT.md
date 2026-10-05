# Public deployment

Check your university/cluster rules before exposing a service from a compute node.

## Full site behind a password (free)

Every page and every path (Fast, Adaptive, Agentic, External, runs, gold, benchmark) for the people
you give the password to. One command on the node:

```sh
# once, without root: the Cloudflare tunnel client
mkdir -p ~/.local/bin && curl -L -o ~/.local/bin/cloudflared \
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x ~/.local/bin/cloudflared

# stop `make demo` first (same ports), then:
SITE_USERS='aditya:<long passphrase>,prof:<another long passphrase>' make public
```

It builds the UI, starts the corpus server, the API (loopback) and `api/public_site.py` on
`127.0.0.1:8080`, opens a Cloudflare quick tunnel to that port only and prints the address
(`https://<words>.trycloudflare.com`). Free: no account, no domain. The address changes on every
start and quick tunnels have no uptime guarantee; the UI falls back to polling where run events do
not stream. Keep it running in `tmux` or `screen`; Ctrl-C stops everything. Logs:
`artifacts/public-full/`.

- Login: HTTP Basic auth for every request. `SITE_USERS` (name:password pairs) or `SITE_PASSWORD`
  (any name); at least 12 characters; 10 failed logins lock a client out for 10 minutes.
- The API, corpus server and models stay on loopback; only the site port is behind the tunnel.
- Anyone with a login can start any run, including long Agentic and External runs on the GPUs, and
  edit gold rows: give logins only to people you trust, and stop the site while benchmark runs need
  the capacity.
- `TUNNEL=none make public` starts the site without a tunnel (for your own tunnel in front of
  port 8080). A stable address needs a named tunnel (a Cloudflare account and a domain) or another
  service; the site itself stays the same.

For anonymous visitors (Fast and Adaptive, CAPTCHA, daily quota, no login) there is the public
gateway, its own repository (public-gateway, decision 0002): it runs against this API on loopback.
