#!/usr/bin/env python3
"""PWA (installable app + offline shell) head tags injected once at startup.

Wires up three static assets that live in `src/static/` (served by Streamlit
at the `/app/static/` URL path once `[server] enableStaticServing = true` is
set in `.streamlit/config.toml`):

- `manifest.json` - name/icons/theme-color/`display: standalone`, which is
  what lets Chrome/Android offer an "Install app" prompt and lets the
  installed app launch without browser chrome.
- `icon-192.png` / `icon-512.png` - the two icon sizes the manifest requires.
- `service-worker.js` - a minimal cache for the static shell only.

Known limitation (see issue #146): Streamlit's page is rendered by a live
Python process and streamed over a websocket, so there is no static HTML
payload that can be safely cached and replayed offline - doing so would risk
serving stale or broken scheduling state. The service worker registered here
is therefore deliberately scoped to just the static assets above (manifest,
icons, itself), not the live scheduling UI; see src/static/service-worker.js
for the caching logic and its scope comment.

The registration script feature-detects `navigator.serviceWorker` and
swallows any registration failure so browsers without service worker support
(or restrictive environments, e.g. some in-app browsers) don't log an
uncaught console error.

KNOWN LIMITATION - verified empirically against the pinned Streamlit release
(streamlit/web/server/app_static_file_handler.py): its static-file route only
sets a real Content-Type for a hard-coded extension whitelist (images, fonts,
pdf/xml/json); anything outside that list - including `.js` - is force-served
as `Content-Type: text/plain`. The Service Worker spec requires a JavaScript
MIME type on the script response, so `service-worker.js` served this way will
be rejected by strict browsers (e.g. Chrome) even though the file's contents
are valid JS. There is no supported app-level workaround (blob:/data: URLs
are rejected outright by `ServiceWorkerContainer.register()`, and no
whitelisted extension maps to a JS MIME type either). This does not affect
`manifest.json` or the icons, which ARE correctly served (`application/json`
and `image/png` respectively) and already provide real install-prompt value.
Fixing this for real requires either an upstream Streamlit change to
`SAFE_APP_STATIC_FILE_EXTENSIONS` or a reverse proxy in front of the app that
rewrites the Content-Type header for this one path - both out of scope here.
The registration call is still made (and still guarded by the try/catch
above) so the fix is a pure infra change once available, not an app rewrite.
"""

from __future__ import annotations

PWA_HEAD_HTML = """
<link rel="manifest" href="./app/static/manifest.json">
<meta name="theme-color" content="#2e7d32">
<link rel="apple-touch-icon" href="./app/static/icon-192.png">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<meta name="apple-mobile-web-app-title" content="Pickleball">
<script>
(function () {
    if ("serviceWorker" in navigator) {
        window.addEventListener("load", function () {
            navigator.serviceWorker
                .register("./app/static/service-worker.js")
                .catch(function () {
                    // Registration can fail in unsupported/restrictive
                    // browsers (e.g. no HTTPS, disabled SW support). The app
                    // itself is fully usable without the service worker, so
                    // this is intentionally swallowed rather than surfaced
                    // as a console error.
                });
        });
    }
})();
</script>
"""


def inject_pwa_manifest(st_module) -> None:
    """Render the PWA manifest/service-worker head tags. Call once per app
    render (idempotent - the browser dedupes repeated <link>/<script> tags
    with the same content, and service worker registration is itself a
    no-op if the worker is already registered and unchanged)."""
    st_module.markdown(PWA_HEAD_HTML, unsafe_allow_html=True)
