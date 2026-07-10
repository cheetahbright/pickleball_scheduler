/**
 * Minimal "app shell" service worker for the Pickleball Scheduler PWA.
 *
 * Scope note (see GitHub issue #146): Streamlit's main page is rendered by a
 * live Python process and streamed over a websocket - there is no static
 * HTML/JSON payload that can be cached and safely replayed offline without
 * risking a stale or broken UI. Trying to cache that traffic here would
 * silently break live scheduling. Instead this worker's job is intentionally
 * narrow: it only caches the handful of *static* files needed for
 * installability (the manifest + icons + itself), and it is registered from
 * inside src/static/, which limits its default service-worker "scope" to the
 * /app/static/ path - it physically cannot intercept requests for the
 * dynamic Streamlit app shell, websocket traffic, or API calls, since those
 * live outside its scope.
 *
 * KNOWN LIMITATION (verified empirically, see src/pwa.py docstring for the
 * full writeup): Streamlit's static-file server force-serves this file with
 * `Content-Type: text/plain` because `.js` is not in its whitelist of
 * extensions eligible for content-type sniffing. Strict browsers (e.g.
 * Chrome) require a JavaScript MIME type on the service worker script
 * response and will refuse to register this file until that's fixed
 * upstream or via a reverse proxy. The registration is still attempted
 * (guarded by a .catch()) since manifest.json and the icons ARE served with
 * correct types and already provide real "Install app" value on their own.
 */

const CACHE_NAME = "pickleball-shell-v1";

// Only ever the small set of static assets served from this same directory.
// Deliberately does NOT include the Streamlit app page itself or any
// websocket/XHR endpoints.
const SHELL_ASSETS = ["./manifest.json", "./icon-192.png", "./icon-512.png", "./service-worker.js"];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) => cache.addAll(SHELL_ASSETS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((names) => Promise.all(names.filter((name) => name !== CACHE_NAME).map((name) => caches.delete(name))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  const isShellAsset = SHELL_ASSETS.some((asset) => url.pathname.endsWith(asset.replace("./", "/")));

  if (!isShellAsset) {
    // Not one of our known static assets (e.g. dynamic app content, the
    // websocket upgrade, API calls) - let it fall straight through to the
    // network untouched. We do not call event.respondWith() here.
    return;
  }

  // Cache-first for the small static shell: fast repeat loads, with a
  // network fallback so updated icons/manifest are eventually picked up
  // once the cache is refreshed on the next "install".
  event.respondWith(
    caches.match(event.request).then((cached) => {
      if (cached) {
        return cached;
      }
      return fetch(event.request).catch(() => cached);
    })
  );
});
