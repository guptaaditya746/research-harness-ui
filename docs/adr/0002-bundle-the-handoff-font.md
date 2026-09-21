# ADR 0002: Bundle Source Serif 4 for the web app

Status: accepted for M0.

The handoff requires Source Serif 4. Its stylesheet is copied without changes,
including the Google Fonts import. The M0 app also ships the Latin WOFF2 files
locally and registers them before React renders. This keeps the typography
available when the browser cannot reach Google Fonts and makes screenshot tests
repeatable on the compute image, which has no system fonts.

The loaded face is the same family and weights specified by the handoff. The
app's own stylesheet points the design-system font variables at that local
face. The original design-system stylesheet remains unchanged.
