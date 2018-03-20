
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer


# ## 1. Imputer() Implementation
# 
# Selection: 
# * Mean
# * Median

# In[4]:


"""
Needs a file i/o function. 
"""
class imputation(object):
    def __init__(self, df, strategy):
        self.df = df
        self.strategy = strategy
        return
    
    def imputer(self):
            df = pd.DataFrame(np.random.randn(15,6))
            # df = pd.read_excel('')

            imp = Imputer(missing_values='NaN', strategy = self.strategy, axis=0)

            # transformed dataset after imputation 
            # new_df = imp.fit_transform(df)
            # print "This is the dataset with missing fields" 
            # print df 
            # print '\n'
            # print "This is the transformed dataset with missing fields filled." 
            # print new_df

            return

    dff = pd.DataFrame(np.random.randn(11,2))
    imputer(dff)


# ##  2. Zeroth Order Categorical Drop

# ### Sample criteria for method selection
# 
# Pseudo code for checking when to drop the column. 
# * If there are roughly 90% (estimate) of minimal required n, then use mean/median imputation. 
# * If there are 50% of minimal required n missing, then use machine learning techniques.
# * If there are 20% of minimal required n missing, then drop the whole row/column. 
# 
# Default assumptions for minimal sample size determination (should be adjustable based on user preference): 
# * 95% confidence level ($\alpha$) is desired across the board for any analysis --> z-score of 1.96
# * margin of error(MoE): +/-5%
# * Population mean is unknown
# * Sample standard deviation ($s$) can be determined by available fields within that column
# 
# Equation a: $ n = (t_{df,\frac{\alpha}{2}} * \frac{s}{MoE})^2$. 
# 
# Equation b (minimal viable percentage): $ p = \frac{count(missing)}{n} $
# 
# **for** every row in the spreadsheet:  
#     count the number of NaN cells in that particular column  
#     **if** (total fields - number of NaN) < n:  
#         perform below drop sequences accordingly  
#     **end**  
# **end**  
#     

# In[ ]:


n = (t * s / MOE)^2


# In[1]:


"""
Needs a file i/o function and user input prompt.
"""

def dropCheck():
    # Implement the above pseudo code

class naCheck():
    # dff = pd.read_excel('')
    
    # check for null in dataset. Returns boolean
    def naCheck(outcome1):
        outcome1 = dff.isnull()
        return outcome1

class dropNArow():
    # df = pd.read_excel('')
    # drops full row that contains NA by default.
    def dropNArow(outcome2): 
        outcome2 = dff.dropna()
        return outcome2

class dropNAcol():
    # df = pd.read_excel('')
    # drops column instead
    def dropNAcol(outcome3):
        outcome3 = dff.dropna(axis = 'columns')
        return outcome3

class fillNAzero():
    # fill NA fields with 0. Can be further customized to fill based on user input. 
    # df = pd.read_excel('')
    def fillNAzero(outcome4):
        outcome4 = dff.fillna()
        return outcome4

    #naCheck(dff)
    #dropNArow(dff)
    dropNAcol(dff)
    #fillNAzero(dff)


# ## PCA

# ## 4. Machine Learning Methods

# ### 4.1 Decision Tree

# Try Pandas dataframe. 

# In[16]:


import pandas as pd
import numpy as np

# Load the sample dataset
data = pd.read_csv('MASTMLsample.csv')

# extracting a sample column with NaN
feature_original = data['Site2_ThirdIonizationEnergy']
target_original = data['Stability']
feat_mod = []
ind_nan = []
target_mod = []

for i in feature_original:
    if i != 'nan':
        feat_mod.append(i)

# array with NaN removed 
feat_mod = feature_original[~np.isnan(feature_original)]

# row index of NaN array
ind_nan = np.where(np.isnan(feature_original))

# remove and reassign input array that has index corresponding to NaN indices
for i in ind_nan:
    print type(i)
    for j in i:
        print j
        target_mod = target_original.pop(j)


#print ind_nan
print target_mod

# target array used for predicting NaN fields
#target = list(set(target_original) - set(target_mod))


# In[6]:


from sklearn.cross_validation import train_test_split as tts

X_train, X_test, y_train, y_test = tts(feature_mod, target_mod, test_size=0.2)

print "Training and testing split was successful."


# In[7]:


# gets r2 score for regression
from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):
    score = r2_score(y_true, y_predict)
    return score 


# In[8]:


from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import ShuffleSplit

def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)
    
    # regressor object
    regressor = DecisionTreeRegressor()
    
    """dictionary for the parameter 'max_depth' with a range from 1 to 10.
       max_depth: how many questions the decision tree algorithm is allowed to ask about the data before making a prediction"""
    params = {"max_depth": range(1, 20)}
    
    # Make a scorer from a performance metric or loss function.
    scoring_fnc = make_scorer(performance_metric)
    
    # the grid search cv object
    grid = GridSearchCV(regressor, params, scoring = scoring_fnc, cv = cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


# In[10]:


# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

new_input = target.values.reshape(-1, 1)

reg.predict(new_input)

#for i, target in enumerate(reg.predict(new_input)):
   # print target

