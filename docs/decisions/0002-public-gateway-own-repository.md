# 0002 The public gateway moves to its own repository

**Context.** The gateway is internet-facing, has its own threat model (docs/PUBLIC_THREAT_MODEL.md), and by
rule imports no harness code: it calls the API over HTTP.

**Decision.** A separate `public-gateway` repository that talks to this API over HTTP.

**Consequences.** About 1,400 lines leave this repository; the gateway's releases and security fixes no longer
ride on harness commits. `make public` here starts the harness side only and points at the gateway's checkout.
