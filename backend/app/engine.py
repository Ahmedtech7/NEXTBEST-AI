"""Reproducible synthetic randomized four-arm experiment and causal outcome models.
Synthetic outcomes are *not* evidence of real bank intervention effectiveness.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

HERE = Path(__file__).resolve().parents[1]
ARTIFACT = HERE / 'artifacts' / 'model.joblib'
SEED = 42
ACTIONS = ['none','reminder','assistance','incentive']
COSTS = np.array([0,.15,2.50,8.00])
STAGES = ['Personal details','Document upload','Income verification','Employment information','Final review']
PRODUCTS = ['Personal loan','Credit card','Auto finance']
FEATURES = ['progress','inactive','sessions','errors','stage_code','product_code','previous_contact']

def sigmoid(v):return 1/(1+np.exp(-np.clip(v,-30,30)))

def generate(n=10000,seed=SEED):
 rng=np.random.default_rng(seed)
 progress=rng.integers(10,99,n)
 inactive=rng.integers(0,15,n)
 sessions=rng.integers(1,13,n)
 errors=rng.integers(0,6,n)
 stage=rng.integers(0,5,n)
 product=rng.integers(0,3,n)
 previous=rng.integers(0,4,n)
 features=np.column_stack((progress,inactive,sessions,errors,stage,product,previous)).astype(float)
 base_logit=-1.25 + .025*(progress-45)-.09*inactive+.075*sessions-.16*errors+.16*stage-.08*(product==2)
 base=sigmoid(base_logit)
 effects=np.column_stack((np.zeros(n),
       .025+.12*(inactive>=3)-.055*(inactive<2)+.03*(stage==1),
       .04+.18*(errors>=2)+.065*(progress<55)-.025*(sessions>8),
       .055+.18*(progress<45)+.05*(product==0)-.055*(inactive>=10)))
 potentials=np.clip(base[:,None]+effects,.02,.98)
 # Randomized four-arm allocation at equal known propensity.
 assigned=rng.integers(0,4,n)
 outcomes=rng.binomial(1,potentials[np.arange(n),assigned])
 df=pd.DataFrame(features,columns=FEATURES)
 df['assigned_action']=assigned;df['observed_completion']=outcomes
 return df,potentials

def make_model():return HistGradientBoostingRegressor(max_iter=95,max_leaf_nodes=16,l2_regularization=8,learning_rate=.075,min_samples_leaf=55,random_state=SEED)

def train(n=10000):
 df,truth=generate(n)
 train_idx,test_idx=train_test_split(np.arange(n),test_size=.25,random_state=SEED,stratify=df.assigned_action)
 X=df[FEATURES].to_numpy(); arms=df.assigned_action.to_numpy(); y=df.observed_completion.to_numpy()
 models=[]
 for action in range(4):
  mask=train_idx[arms[train_idx]==action]
  model=make_model();model.fit(X[mask],y[mask]);models.append(model)
 preds=np.column_stack([np.clip(model.predict(X[test_idx]),.01,.99) for model in models]); real=truth[test_idx]
 # Cost penalty represents an illustrative utility conversion, NOT a monetary business case.
 cost_pp=COSTS*.004
 policy=np.argmax(preds-cost_pp,axis=1)
 baseline=real[:,0]
 oracle=np.argmax(real-cost_pp,axis=1)
 real_policy=real[np.arange(len(test_idx)),policy]
 # Replay-based off-policy IPS estimate for randomized assignment with propensity 1/4.
 observed=y[test_idx]; allocated=arms[test_idx]
 ips=np.mean((allocated==policy)*observed*4)
 control_ips=np.mean((allocated==0)*observed*4)
 metrics={'dataset_size':n,'training_size':len(train_idx),'test_size':len(test_idx),
          'randomization':'uniform random assignment; propensity = 0.25 per arm',
          'oracle_policy_completion':round(float(np.mean(real[np.arange(len(test_idx)),oracle])),4),
          'policy_completion_oracle_truth':round(float(np.mean(real_policy)),4),
          'control_completion_oracle_truth':round(float(np.mean(baseline)),4),
          'policy_incremental_uplift_pp':round(float(np.mean(real_policy-baseline)*100),2),
          'ips_policy_completion':round(float(ips),4),'ips_control_completion':round(float(control_ips),4),
          'ips_incremental_uplift_pp':round(float((ips-control_ips)*100),2),
          'counterfactual_rmse':round(float(np.sqrt(np.mean((preds-real)**2))),4),
          'action_counts':{ACTIONS[i]:int(np.sum(arms==i)) for i in range(4)},
          'note':'Synthetic randomized outcomes only. No real-world causal efficacy is established.'}
 artifact={'models':models,'metrics':metrics,'features':FEATURES,'seed':SEED,'version':'1.0'}
 ARTIFACT.parent.mkdir(parents=True,exist_ok=True);joblib.dump(artifact,ARTIFACT)
 return artifact

def load():
 if not ARTIFACT.exists():raise RuntimeError('Missing model artifact; run python -m app.train first')
 return joblib.load(ARTIFACT)

def predict(artifact,features):
 X=np.asarray(features,dtype=float).reshape(-1,len(FEATURES))
 return np.column_stack([np.clip(m.predict(X),.01,.99) for m in artifact['models']])

def customer_rows(artifact,count=120):
 df,_=generate(); sample=df.iloc[:count].copy()
 predicted=predict(artifact,sample[FEATURES].to_numpy())
 best=np.argmax(predicted-COSTS*.004,axis=1)
 result=[]
 for i,(_,r) in enumerate(sample.iterrows()):
  scores=[round(float(x*100),1) for x in predicted[i]]
  b=int(best[i]);base=scores[0]
  result.append({'id':f'CUST-{i+1001:05d}','name':f'Synthetic customer {i+1001}',
   'initials':str(i+1001)[-2:],'stage':STAGES[int(r.stage_code)],'product':PRODUCTS[int(r.product_code)],
   'baseline':base,'probabilities':scores,'best':b,'progress':int(r.progress),
   'inactive':int(r.inactive),'sessions':int(r.sessions),'errors':int(r.errors),
   'previous_contact':int(r.previous_contact),'risk':'High' if base<40 else 'Medium' if base<67 else 'Low'})
 return result

if __name__=='__main__':
 print(json.dumps(train()['metrics'],indent=2))
