from mastml.models import HostedModel
import pandas as pd
 
X_test = pd.read_csv('test.csv')
 
model = HostedModel(container_name='rjacobs3/diffusion_test:updated8')
 
preds = model.predict(X_test)
 
print(preds)
