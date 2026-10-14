#!/usr/bin/env python3
"""PWA (installable app) head tags added once at startup.

Wires up three static assets that live in `src/static/` (served by Streamlit
at the `/app/static/` URL path once `[server] enableStaticServing = true` is
set in `.streamlit/config.toml`; Streamlit serves the `static/` folder that
sits beside the script given to `streamlit run`, so an entry script outside
`src/` needs a `static/` of its own):

- `manifest.json` - name/icons/theme-color/`display: standalone`, which is
  what lets Chrome/Android offer an "Install app" prompt and lets the
  installed app launch without browser chrome.
- `icon-192.png` / `icon-512.png` - the two icon sizes the manifest requires.
- `service-worker.js` - a minimal cache for those static files only. It
  gives the app no offline support (see the scope note below).

Browsers read the manifest link, theme-color and apple-* tags only from the
page's <head>, and Streamlit renders app content into <body>. So the tags are
not rendered as HTML: `inject_pwa_manifest` renders a script through
`st.html(..., unsafe_allow_javascript=True)`, which Streamlit runs in the page
itself (not in an iframe), and the script creates the tags in
`document.head`. Earlier versions rendered the tags with `st.markdown`; they
landed in a <div> in the body, where browsers ignore them, and the
registration <script> never ran. A host that wraps the app in an iframe of
its own page (Streamlit Community Cloud serves the app itself at `/~/+/`)
keeps the tags in that frame, and browsers take the install manifest only
from the top-level page, so such a host probably still shows no install
prompt. The script does not write into the host's own page.

Known limitation (see issue #146): Streamlit's page is rendered by a live
Python process and streamed over a websocket, so there is no static HTML
payload that can be safely cached and replayed offline - doing so would risk
serving stale or broken scheduling state. The service worker registered here
is therefore deliberately scoped to just the static assets above (manifest,
icons, itself), not the live scheduling UI; see src/static/service-worker.js
for the caching logic and its scope comment. That scope (`/app/static/`) also
means it never controls the app page: in a local Chromium check with the
worker served as JavaScript, it registered and cached its four files, but
`navigator.serviceWorker.controller` stayed null on the app page, even after a
reload. So even when registered it adds no offline support, and installing
the app does not need it.

The registration script feature-detects `navigator.serviceWorker` and
swallows any registration failure so browsers without service worker support
(or restrictive environments, e.g. some in-app browsers) don't log an
uncaught console error.

KNOWN LIMITATION - verified empirically against the pinned Streamlit release
(streamlit/web/server/app_static_file_handler.py): its static-file route only
sets a real Content-Type for a hard-coded extension whitelist (images, fonts,
pdf/xml/json); anything outside that list - including `.js` - is force-served
as `Content-Type: text/plain`. The Service Worker spec requires a JavaScript
MIME type on the script response, so `service-worker.js` served this way is
rejected by strict browsers (e.g. Chrome, which also logs a console error even
when the rejection is caught) even though the file's contents are valid JS.
There is no supported app-level workaround (blob:/data: URLs are rejected
outright by `ServiceWorkerContainer.register()`, and no whitelisted extension
maps to a JS MIME type either). This does not affect `manifest.json` or the
icons, which ARE correctly served (`application/json` and `image/png`
respectively) and are all Chrome needs to offer the install prompt: in a
local Chromium check it reported no installability errors and fired
`beforeinstallprompt` with no service worker registered.
Fixing this for real requires either an upstream Streamlit change to
`SAFE_APP_STATIC_FILE_EXTENSIONS` or a reverse proxy in front of the app that
rewrites the Content-Type header for this one path - both out of scope here.
The script therefore first asks for the worker with a HEAD request and
registers it only when it comes back as JavaScript, so nothing is logged
meanwhile. Fixing the Content-Type would only let the worker register; per
the scope note above, it would still bring the app page no offline benefit.
"""

from __future__ import annotations

import json

# (tag, attributes) for each element the script adds to document.head.
PWA_HEAD_TAGS = (
    ("link", {"rel": "manifest", "href": "./app/static/manifest.json"}),
    ("meta", {"name": "theme-color", "content": "#2e7d32"}),
    ("link", {"rel": "apple-touch-icon", "href": "./app/static/icon-192.png"}),
    ("meta", {"name": "apple-mobile-web-app-capable", "content": "yes"}),
    # Chrome warns that the apple- name above is deprecated unless this is present too.
    ("meta", {"name": "mobile-web-app-capable", "content": "yes"}),
    ("meta", {"name": "apple-mobile-web-app-status-bar-style", "content": "default"}),
    ("meta", {"name": "apple-mobile-web-app-title", "content": "Pickleball"}),
)

SERVICE_WORKER_URL = "./app/static/service-worker.js"

# st.html sanitizes its body with DOMPurify, which drops any element whose
# text contains "<" followed by a letter, "/" or "!" - so this script must
# never contain one, not even in a comment.
PWA_HEAD_HTML = f"""
<script>
(function () {{
    var head = document.head;
    // Streamlit can remount this element, which runs the script again.
    if (!head.querySelector('link[rel="manifest"]')) {{
        {json.dumps(PWA_HEAD_TAGS)}.forEach(function (tag) {{
            var element = document.createElement(tag[0]);
            Object.keys(tag[1]).forEach(function (name) {{
                element.setAttribute(name, tag[1][name]);
            }});
            head.appendChild(element);
        }});
    }}
    if ("serviceWorker" in navigator) {{
        // The page's load event has already fired by the time Streamlit
        // renders this, so register now rather than waiting for it. Streamlit
        // serves .js as text/plain (see the module docstring), which the
        // browser rejects with a console error, so only register once the
        // worker is served as JavaScript.
        var workerUrl = {json.dumps(SERVICE_WORKER_URL)};
        fetch(workerUrl, {{ method: "HEAD" }})
            .then(function (response) {{
                var type = response.headers.get("Content-Type") || "";
                if (response.ok && type.indexOf("javascript") !== -1) {{
                    return navigator.serviceWorker.register(workerUrl);
                }}
            }})
            .catch(function () {{
                // Registration can fail in unsupported/restrictive browsers
                // (e.g. no HTTPS, disabled SW support). The app itself is
                // fully usable without the service worker.
            }});
    }}
}})();
</script>
"""


def inject_pwa_manifest(st_module) -> None:
    """Render the script that adds the PWA tags to <head> and registers the
    service worker. Call once per app render: the script runs when Streamlit
    mounts the element, skips the tags if <head> already has them, and
    service worker registration is itself a no-op if the worker is already
    registered and unchanged."""
    st_module.html(PWA_HEAD_HTML, unsafe_allow_javascript=True)
