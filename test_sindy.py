#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import os
import pyreadr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import torch
# import lightgbm as lgb
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# import tensorflow as tf

sns.set()


# Sindy packages

# In[3]:


import numpy as np
import pysindy as ps


# Setup training and testing data

# In[4]:


train_normal_path = './Dataset/TEP_FaultFree_Training.RData'
train_faulty_path = './Dataset/TEP_Faulty_Training.RData'

test_normal_path = './Dataset/TEP_FaultFree_Testing.RData'
test_faulty_path = './Dataset/TEP_Faulty_Testing.RData'


train_df = pyreadr.read_r(train_normal_path)['fault_free_training']    # (250000, 55)
test_df = pyreadr.read_r(test_faulty_path)['faulty_testing']


# simulationRun = 1 | faultNumber = 0 (faultless)  dataset 설정
# 샘플이 총 500개있음. 500 샘플을 시간축으로 간주함.

# In[6]:


feature_names = train_df.columns[3:]
train_normal_df = train_df[(train_df.faultNumber==0)&(train_df.simulationRun.isin(range(11)))][feature_names]   # (10000, 52)
X_np = train_normal_df.to_numpy()

dt = 1
n_states = len(train_normal_df.columns)
t = len(train_normal_df.index)
print(f'feature_names:{feature_names}')


# Scaler

# In[ ]:


scaler = preprocessing.MinMaxScaler()
train_df_scaled = pd.DataFrame(scaler.fit_transform(X = train_normal_df))
train_df_scaled
X_np = train_df_scaled.to_numpy()


# truncated SVD
# 

# In[6]:


SVD_RANK = 15
# get SVD rank
X1 = X_np[:-1, :]   # (499, 52)
X2 = X_np[1:, :]

U, s, Vh = np.linalg.svd(X1.T, full_matrices=False)



# differentiation submodule

# In[7]:


differentiation_method = ps.FiniteDifference(order=2)


# Feature_library submodule

# In[8]:


feature_library = ps.PolynomialLibrary(degree=3)

# Note: We could instead call ps.feature_library.PolynomialLibrary(degree=3)


# Next we select which optimizer should be used.

# In[9]:


optimizer = ps.STLSQ(threshold=0.2)

# Note: We could instead call ps.optimizers.STLSQ(threshold=0.2)


# Finally, we bring these three components together in one `SINDy` object.

# In[10]:


model = ps.SINDy(
    differentiation_method=differentiation_method,
    feature_library=feature_library,
    optimizer=optimizer,
    feature_names=feature_names,
)


# Following the `scikit-learn` workflow, we first instantiate a `SINDy` class object with the desired properties, then fit it to the data in separate step.

# In[11]:


model.fit(X_np, t=dt)


# We can inspect the governing equations discovered by the model and check whether they seem reasonable with the `print` function.

# In[ ]:


model.print()

