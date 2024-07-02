#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import pandas as pd


# In[2]:


train_data = pd.read_csv("train.csv")
train_data.head()
test_data = pd.read_csv("test.csv")


# In[3]:


def concat_df(train_data, test_data):
    # Returns a concatenated df of training and test set
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)

def divide_df(all_data):
    # Returns divided dfs of training and test set
    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis=1)

all_data = concat_df(train_data, test_data)


# In[4]:


from sklearn.ensemble import RandomForestClassifier


# In[5]:


sums = {
    "male": {
        1: 0,
        2: 0,
        3: 0
    },
    "female": {
        1: 0,
        2: 0,
        3: 0
    }
}
cnts = {
    "male": {
        1: 0,
        2: 0,
        3: 0
    },
    "female": {
        1: 0,
        2: 0,
        3: 0
    }
}

for i in range(len(train_data)):
    if np.isnan(train_data["Age"][i]):
        continue
    sums[train_data["Sex"][i]][train_data["Pclass"][i]] += train_data["Age"][i]
    cnts[train_data["Sex"][i]][train_data["Pclass"][i]] += 1 

for i in range(len(all_data)):
    if np.isnan(all_data["Age"][i]):
        tmp = sums[all_data["Sex"][i]][all_data["Pclass"][i]] / cnts[all_data["Sex"][i]][all_data["Pclass"][i]]
        all_data["Age"][i] = tmp


# In[6]:


y = train_data["Survived"]


# In[7]:


X, X_test = divide_df(all_data)


# In[8]:


X_test


# In[9]:


features = ["Age", "Embarked", "Parch", "Pclass", "Sex", "SibSp"]


# In[10]:


X = pd.get_dummies(X[features])
X_test = pd.get_dummies(X_test[features])


# In[11]:


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
kfold = StratifiedKFold(n_splits=10)


# In[12]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

ada = AdaBoostClassifier()
knn = KNeighborsClassifier()
mlp = MLPClassifier()
svm = SVC(probability=True)
rfc = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)


# In[13]:


from sklearn.model_selection import GridSearchCV

ada_param_grid = {"algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[30, 50, 70, 100],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}

gsadaDTC = GridSearchCV(ada,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs = 8)
gsadaDTC.fit(X, y)
ada_best = gsadaDTC.best_estimator_

print("finished adaboost")
# In[14]:


svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}

best_svm = GridSearchCV(svm,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs = 8)
best_svm.fit(X, y)
svc_best = best_svm.best_estimator_

print("finished svc")
# In[15]:


knn_param_grid = {'n_neighbors' : [5,7,9,11,13,15],
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']}

best_knn = GridSearchCV(knn,param_grid = knn_param_grid, cv=kfold, scoring="accuracy", n_jobs = 8)
best_knn.fit(X, y)
knn_best = best_knn.best_estimator_


print("finished knn")
# In[16]:


n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 5)]
# Number of features to consider at every split
max_features = ['log2', 'sqrt', None]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 50, num = 5)]
max_depth.append(None)
# Create the random grid
rfc_param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth}
best_rfc = GridSearchCV(rfc,param_grid = rfc_param_grid, cv=kfold, scoring="accuracy", n_jobs = 8)
best_rfc.fit(X, y)
rfc_best = best_rfc.best_estimator_


print("finished rfc")
# In[17]:


import xgboost as xgb
from xgboost import XGBClassifier 

xgb_param_grid = {
    
    'n_estimators': [100, 500, 900],
    'max_depth': [1, 3, 5, 7],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 1.5]
}

# Create an XGBoost model
xgb_model = XGBClassifier()

# Use GridSearchCV to find the optimal hyperparameters
best_xgb = GridSearchCV(xgb_model, param_grid=xgb_param_grid, cv=kfold, scoring="accuracy", n_jobs=8)
best_xgb.fit(X, y)
xgb_best = best_xgb.best_estimator_


print("finished xgb")
# In[18]:


print(cross_val_score(rfc_best, X, y, scoring = "accuracy", cv = kfold, n_jobs=8).mean())
print(cross_val_score(svc_best, X, y, scoring = "accuracy", cv = kfold, n_jobs=8).mean())
print(cross_val_score(ada_best, X, y, scoring = "accuracy", cv = kfold, n_jobs=8).mean())
print(cross_val_score(knn_best, X, y, scoring = "accuracy", cv = kfold, n_jobs=8).mean())
print(cross_val_score(xgb_best, X, y, scoring = "accuracy", cv = kfold, n_jobs=8).mean())


# In[19]:


from sklearn.ensemble import VotingClassifier
votingC = VotingClassifier(estimators=[('xgb', xgb_best), ('rfc', rfc_best), ('svc', svc_best), ('ada',ada_best),('knn',knn_best)], voting='soft', n_jobs=8)
#votingC = VotingClassifier(estimators=[('rfc', rfc)], voting='soft', n_jobs=4)

votingC = votingC.fit(X, y)


# In[20]:


print(cross_val_score(votingC, X, y, scoring = "accuracy", cv = kfold, n_jobs=8).mean())


# In[21]:


med_fare = all_data.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
med_fare


# In[22]:


#X_test["Fare"] = X_test["Fare"].fillna(med_fare)


# In[23]:


X_test.info()


# In[24]:


predictions = votingC.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")


# In[ ]:




