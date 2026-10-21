import os
from contextlib import asynccontextmanager
from functools import lru_cache
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .engine import load,customer_rows,predict,FEATURES,ACTIONS,COSTS,STAGES,PRODUCTS

@lru_cache(maxsize=1)
def artifact():return load()

@asynccontextmanager
async def lifespan(app):
 artifact()
 yield

app=FastAPI(title='NEXTBEST AI — Synthetic Uplift API',version='1.0',lifespan=lifespan)
origins=[s.strip() for s in os.getenv('FRONTEND_ORIGINS','http://localhost:5173,http://localhost:5175,http://127.0.0.1:5173,http://127.0.0.1:5175').split(',') if s.strip()]
app.add_middleware(CORSMiddleware,allow_origins=origins,allow_methods=['GET','POST'],allow_headers=['Content-Type'])

class PredictRequest(BaseModel):
 progress:int=Field(ge=0,le=100)
 inactive:int=Field(ge=0,le=90)
 sessions:int=Field(ge=0,le=100)
 errors:int=Field(ge=0,le=30)
 stage_code:int=Field(ge=0,le=4)
 product_code:int=Field(ge=0,le=2)
 previous_contact:int=Field(ge=0,le=25)

@app.get('/api/health')
def health():return {'status':'ok','model':'trained-synthetic-4-arm-t-learner','data':'synthetic'}

@app.get('/api/dashboard')
def dashboard():
 a=artifact();rows=customer_rows(a)
 return {'customers':rows,'metrics':a['metrics'],'actions':[
  {'id':v,'label':label,'short':short,'color':color,'cost':float(COSTS[i])} for i,(v,label,short,color) in enumerate(zip(
   ACTIONS,['No action','Personalized reminder','Offer assistance','Provide incentive'],
   ['No action','Reminder','Assistance','Incentive'],['#7584af','#9b6dff','#53a2ff','#f4b74c']))],
  'model_mode':'trained_synthetic','dataset_description':'10,000 simulated customers; randomized logged interventions; held-out causal evaluation'}

@app.post('/api/predict')
def predict_customer(body:PredictRequest):
 vec=[getattr(body,f) for f in FEATURES]
 scores=predict(artifact(),vec)[0]
 best=int(np.argmax(scores-COSTS*.004))
 return {'probabilities':[round(float(v*100),2) for v in scores], 'recommended_action':ACTIONS[best],
  'uplift_pp':round(float((scores[best]-scores[0])*100),2),
  'disclosure':'Model estimates trained and evaluated exclusively on synthetic randomized data.'}

@app.get('/api/metrics')
def metrics():return artifact()['metrics']
