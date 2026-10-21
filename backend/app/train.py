from .engine import train
import json
if __name__=="__main__":
 print(json.dumps(train()["metrics"], indent=2))
