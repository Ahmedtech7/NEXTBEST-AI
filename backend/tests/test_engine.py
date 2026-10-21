from app.engine import generate,train,predict,FEATURES

def test_synthetic_randomization():
 d,p=generate(2000)
 assert len(d)==2000 and p.shape==(2000,4)
 assert d.assigned_action.nunique()==4
 assert ((p>0)&(p<1)).all()

def test_predictions():
 model=train(2200)
 df,_=generate(4)
 result=predict(model,df[FEATURES].to_numpy())
 assert result.shape==(4,4)
 assert ((result>=0)&(result<=1)).all()
