
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer


# In[6]:


### imputer implementation 
# random dataset for testing
df = pd.DataFrame(np.random.randn(10,6))
# df = pd.read_excel('')

# insert a few NaN values
df.iloc[3,4] = np.nan
df.iloc[6,2] = np.nan
df.iloc[3:5,2] = np.nan
imp = Imputer(missing_values='NaN', strategy="median", axis=0)

# transformed dataset after imputation 
new_df = imp.fit_transform(df)
print "This is the dataset with missing fields" 
print df 
print '\n'
print "This is the transformed dataset with missing fields filled." 
print new_df


# In[26]:


### dropping and filling observations 
# check for null in dataset. Returns boolean
#df.isnull()

# drops full row that contains NA by default.
#df.dropna()

# drops column instead
#df.dropna(axis = 'columns')

# fill NA values
df.fillna(0)

