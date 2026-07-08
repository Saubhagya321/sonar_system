# =============================================================================
# Sonar Frontend (Vite + React 18 + TypeScript) — nginx-only image
#
# The corporate network on this host resets TLS to registry.npmjs.org, so
# building the bundle inside the container isn't currently possible. Instead
# the local dist/ produced by `npm run build` is baked directly into the
# nginx image. VITE_API_URL is baked in at build time by Vite (default in
# .env: http://10.90.90.14/api) — rebuild the bundle locally after changing
# it or after any source change:
#
#   npm run build
#   docker compose build frontend
#
# nginx.conf provides SPA fallback so react-router deep links resolve.
# =============================================================================
FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY dist /usr/share/nginx/html

EXPOSE 80

# nginx:alpine ships with neither wget nor curl; BusyBox nc is available and
# gives a genuine TCP probe of the listener.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD nc -z 127.0.0.1 80 || exit 1

CMD ["nginx", "-g", "daemon off;"]
