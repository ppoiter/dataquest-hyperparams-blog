
# coding: utf-8

# In[1]:


# working for hyperparameters blog piece for dataquest - use UCI bank dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import scipy as sp
from timeit import default_timer as timer
from sklearn.metrics import confusion_matrix

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


bank = pd.read_csv('bank.csv')
bank.head()


# In[3]:


bank = pd.get_dummies(bank, drop_first=True)


# In[4]:


bank.head()


# In[5]:


cutoff = bank.shape[0] * 0.8
train = bank.sample(frac = 0.8)
test = bank.loc[~bank.index.isin(train.index)]


# In[6]:


X = train.drop('y_yes', axis = 1)
y = train.y_yes

X.shape, y.shape


# In[7]:


columns = train.columns.drop('y_yes')
columns


# In[8]:


train['y_yes'].value_counts()


# In[9]:


test['y_yes'].value_counts()


# In[10]:


clf = RandomForestClassifier(max_depth=2, random_state=0, class_weight='balanced')
clf.fit(X,y)
predictions = clf.predict(test[columns])


# In[11]:


true_pos = 0
false_pos = 0
true_neg = 0
false_neg = 0

for i, val in enumerate(predictions):
    if val == 1 and test['y_yes'].iloc[i] ==1:
        true_pos += 1
    elif val == 1 and test['y_yes'].iloc[i] ==0:
        false_pos += 1
    elif val == 0 and test['y_yes'].iloc[i] ==0:
        true_neg += 1
    elif val == 0 and test['y_yes'].iloc[i] ==1:
        false_neg += 1  
        
print(true_pos)
print(false_pos)
print(true_neg)
print(false_neg)


# In[12]:


report = classification_report(test['y_yes'], predictions, digits=5)
print(report)


# In[13]:


correct_count = 0
false_count = 0

for i, val in enumerate(predictions):
    if val == test['y_yes'].iloc[i]:
        correct_count += 1
    else:
        false_count += 1
        
print(correct_count)
print(false_count)


# In[14]:


param_grid = {'n_estimators':[2, 5, 10, 15, 50, 100, 200], 'max_depth':[3, 5, 10, 15, 25, 40, 75]}


# In[16]:


start = timer()
rndm_f = RandomForestClassifier(class_weight='balanced', random_state=2)
clf = GridSearchCV(rndm_f, param_grid, cv = 5)
clf.fit(X,y)
#predictions = clf.predict(test[columns])
predictions = (clf.predict_proba(test[columns])[:,1] >= 0.45).astype(bool)
end = timer()
print('Grid Search took', end - start, 'seconds')


# In[17]:


print (clf.best_params_)


# In[18]:


report = classification_report(test['y_yes'], predictions, digits=5)
print(report)


# In[19]:


print (clf.best_score_)


# In[20]:


true_pos = 0
false_pos = 0
true_neg = 0
false_neg = 0

for i, val in enumerate(predictions):
    if val == 1 and test['y_yes'].iloc[i] ==1:
        true_pos += 1
    elif val == 1 and test['y_yes'].iloc[i] ==0:
        false_pos += 1
    elif val == 0 and test['y_yes'].iloc[i] ==0:
        true_neg += 1
    else:
        false_neg += 1  
        
print('True positives:', true_pos)
print('')
print('False positives:', false_pos)
print('')
print('True negatives:', true_neg)
print('')
print('False negatives:', false_neg)


# In[21]:


correct_count = 0
false_count = 0

for i, val in enumerate(predictions):
    if val == test['y_yes'].iloc[i]:
        correct_count += 1
    else:
        false_count += 1
        
print('Correct predictions:',correct_count)
print('')
print('False predictions', false_count)


# # Random search

# In[22]:


param_dist = {'n_estimators':[2, 5, 10, 15, 50, 100, 200], 'max_depth':[3, 5, 10, 15, 25, 40, 75]}


# In[23]:


start = timer()

rndm_f = RandomForestClassifier(random_state=2)
clf = RandomizedSearchCV(rndm_f, param_dist, cv = 5)
clf.fit(X,y)
#predictions = clf.predict(test[columns])

predictions = (clf.predict_proba(test[columns])[:,1] >= 0.45).astype(bool)

end = timer()
print('Random Search took', end - start, 'seconds')


# In[24]:


print (clf.best_params_)


# In[25]:


report = classification_report(test['y_yes'], predictions, digits=5)
print(report)


# In[26]:


print (clf.best_score_)


# In[27]:


true_pos = 0
false_pos = 0
true_neg = 0
false_neg = 0

for i, val in enumerate(predictions):
    if val == 1 and test['y_yes'].iloc[i] ==1:
        true_pos += 1
    elif val == 1 and test['y_yes'].iloc[i] ==0:
        false_pos += 1
    elif val == 0 and test['y_yes'].iloc[i] ==0:
        true_neg += 1
    else:
        false_neg += 1  
        
print('True positives:', true_pos)
print('')
print('False positives:', false_pos)
print('')
print('True negatives:', true_neg)
print('')
print('False negatives:', false_neg)


# In[28]:


correct_count = 0
false_count = 0

for i, val in enumerate(predictions):
    if val == test['y_yes'].iloc[i]:
        correct_count += 1
    else:
        false_count += 1
        
print('Correct predictions:',correct_count)
print('')
print('False predictions', false_count)

