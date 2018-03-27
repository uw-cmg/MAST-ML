
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer


# ## 1. Imputer() Implementation

# In[17]:


class imputation(object):
    def __init__(self, dataframe, strategy):
        self.dataframe = dataframe
        self.strategy = strategy
    
    def imputer(self):
            # axis 0 = impute by column 
            # axis 1 = impute by row
            # strategy takes "mean", "median", or "most_frequent" 
            imp = Imputer(missing_values='NaN', strategy=self.strategy, axis=0)

            # transformed dataset after imputation 
            new_df = imp.fit_transform(self.dataframe)

            return self


# ##  2. Zeroth Order Categorical Drop

# In[21]:


class drop(object):
    def __init__(self, dataframe, axis, how, method, value):
        self.dataframe = dataframe
        self.axis = axis
        self.how = how
        self.method = method
        self.value = value

        # check for null in dataset. Returns boolean
        def naCheck(outcome1):
            outcome1 = self.dataframe.isnull()
            return outcome1

        # drops row
        def dropNArow(outcome2): 
            # how --> any : if any NA values are present, drop that label
            # how --> all : if all values are NA, drop that label
            outcome2 = self.dataframe.dropna(axis=1, how=self.how, thresh=None, subset=None, inplace=False)
            return outcome2

        # drops column
        def dropNAcol(outcome3):
            # any : if any NA values are present, drop that label
            # all : if all values are NA, drop that label
            outcome3 = self.dataframe.dropna(axis=0, how=self.how, thresh=None, subset=None, inplace=False)
            return outcome3

        # value : scalar or dict
        # Value to use to fill holes (e.g. 0), alternately a dict of values specifying which value to use for each column (columns not in the dict will not be filled)
        def fillNAzero(outcome4):
            outcome4 = self.dataframe.fillna(value=self.value, method=self.method, axis=self.axis, inplace=False, limit=None, downcast=None)
            return outcome4


# ## 3. PCA

# In[ ]:


class pca(object):
    def __init__(self, dataframe, x_features, y_feature):
        self.dataframe = dataframe
        self.x_features = x_features
        self.y_feature = y_feature
        
    def principal_component_analysis(self):
        pca = PCA(n_components=len(self.x_features), svd_solver='auto')
        Xnew = pca.fit_transform(X=self.dataframe[self.x_features])
        dataframe = DataframeUtilities().array_to_dataframe(array=Xnew)
        dataframe = FeatureIO(dataframe=dataframe).add_custom_features(features_to_add=[self.y_feature], data_to_add=self.dataframe[self.y_feature])
        return dataframe


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

