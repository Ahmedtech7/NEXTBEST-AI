# NEXTBEST AI — V4 Decision Intelligence

Production-oriented portfolio proof of concept with **synthetic data only**. Four-arm T-Learner (scikit-learn HistGradientBoostingRegressor), FastAPI and React JSX. Nothing here performs banking operations.

## Run locally

Backend (terminal 1):

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m app.train
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Frontend (terminal 2, from project root):

```bash
npm install
npm run dev -- --port 5188 --strictPort
```

Browse http://localhost:5188. Run Python tests with `cd backend && python -m pytest -q`. For hosted operation see [DEPLOYMENT.md](DEPLOYMENT.md).

## V4 additions

- Scenario Lab allows editing progress, inactivity, sessions, errors and previous contacts, then sends actual `/api/predict` requests to the trained model.
- Model Performance Lab displays held-out synthetic metrics returned from FastAPI, including oracle-simulation policy uplift, IPS uplift and RMSE.
- Decision transparency compares all four model outputs. **This is not SHAP** or feature-level causal attribution.
- Retry connection control and production frontend/backend configuration for Vercel and Render.
- English and Arabic (light Najdi tone) support from V3 is preserved. Newer V4 technical labels remain in English in some areas.

## Important limitations

- Dataset and randomized treatments are entirely synthetic. Observed model outcomes are not evidence of banking impact.
- Model estimates are predictive counterfactual approximations on simulator data; feature edits are not causal interventions.
- Vercel/Render config is provided, but no live deployment is claimed.
- Validate the frontend production build and complete browser smoke testing before treating the deployment as verified.
