# NEXTBEST AI V4 — deployment

## Preflight

From repository root: `npm install && npm run build`. From `backend`: `python -m pip install -r requirements.txt && python -m app.train && python -m pytest -q` (install pytest separately if needed). Keep the trained artifact generated at build time.

## Render backend

1. Push project to GitHub (do not commit secrets or virtual environments).
2. Create Render Python service, choose `backend` as root, or use provided `render.yaml`.
3. Build: `pip install -r requirements.txt && python -m app.train`. Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
4. Set `FRONTEND_ORIGINS=https://YOUR-APP.vercel.app` to the actual approved production frontend origin.
5. Confirm `https://YOUR-API.onrender.com/api/health` and `/api/dashboard` return HTTP 200. On sleeping/free hosts allow cold start.

## Vercel frontend

1. Import same GitHub repo, root directory repository root, framework Vite.
2. Build command `npm run build`; output `dist`.
3. Set `VITE_API_BASE_URL=https://YOUR-API.onrender.com` (no trailing slash). This is a public URL, **not** a secret.
4. Deploy; verify on public URL that Model Analytics shows metrics and scenario lab reaches the hosted API.
5. If CORS errors appear, update Render `FRONTEND_ORIGINS` to match the Vercel origin exactly and redeploy.

## Notes

- Sample data are synthetic and predictions are NOT validated on bank customers.
- No banking actions are executed. API is public: apply provider-level abuse limits before high-traffic launch.
- Browser demo mode is not equivalent to successful API integration.
- Inspect logs and test end-to-end before publicly describing deployment as complete.
