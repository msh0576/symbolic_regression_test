#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import sklearn
from sklearn.metrics import mean_squared_error, r2_score
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import warnings
import torch
import torch.nn as nn
import sys
import pysindy as ps
from torch.utils.data import Dataset, DataLoader, TensorDataset
from scipy.integrate import odeint


np.random.seed(34)
warnings.filterwarnings('ignore')

sindyauto_path = '/home/sihoon/works/HAVOK/SindyAutoencoders_master/src'
if sindyauto_path not in sys.path:
    sys.path.insert(0, sindyauto_path)
from sindy_utils import library_size
from training import train_network
import tensorflow as tf


# In[ ]:





# Data preparation

# In[2]:


index_names = ['unit_number', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['s_{}'.format(i+1) for i in range(0,21)]
col_names = index_names + setting_names + sensor_names


dataset_path = './../Dataset/CMaps'

dftrain = pd.read_csv(os.path.join(dataset_path, 'train_FD001.txt'),sep='\s+',header=None,index_col=False,names=col_names)
dfvalid = pd.read_csv(os.path.join(dataset_path, 'test_FD001.txt'),sep='\s+',header=None,index_col=False,names=col_names)
y_valid = pd.read_csv(os.path.join(dataset_path, 'RUL_FD001.txt'),sep='\s+',header=None,index_col=False,names=['RUL'])
dfvalid.shape

train = dftrain.copy()
valid = dfvalid.copy()

print('Shape of the train dataset : ',train.shape)
print('Shape of the validation dataset : ',valid.shape)
print('Percentage of the validation dataset : ',len(valid)/(len(valid)+len(train)))


# Max time cycle found for each unit

# In[3]:


max_time_cycles=train[index_names].groupby('unit_number').max()
max_time_cycles
plt.figure(figsize=(20,50))
ax=max_time_cycles['time_cycles'].plot(kind='barh',width=0.8, stacked=True,align='center')
plt.title('Turbofan Engines LifeTime',fontweight='bold',size=30)
plt.xlabel('Time cycle',fontweight='bold',size=20)
plt.xticks(size=15)
plt.ylabel('unit',fontweight='bold',size=20)
plt.yticks(size=15)
plt.grid(True)
plt.tight_layout()
plt.show()


# Add RUL comun to the data
# RUL corresponds to the remining time cycles for each unit before it fails

# In[4]:


def add_RUL_column(df):
    '''
    RUL: for each unit_number, 'max_time_cycles' denotes end of the life time.
        Thus, RUL = 'max_time_cycles' - each currnet time 'time_cycles'
    '''
    train_grouped_by_unit = df.groupby(by='unit_number') 
    max_time_cycles = train_grouped_by_unit['time_cycles'].max() 
    # print(max_time_cycles)
    # print(max_time_cycles.to_frame(name='max_time_cycle'))
    merged = df.merge(max_time_cycles.to_frame(name='max_time_cycle'), left_on='unit_number',right_index=True)
    # print(merged[['unit_number', 'max_time_cycle', 'time_cycles']])
    merged["RUL"] = merged["max_time_cycle"] - merged['time_cycles']
    merged = merged.drop("max_time_cycle", axis=1) 
    return merged


# In[5]:


train = add_RUL_column(train)
train[['unit_number','RUL']]


# In[6]:


#Rul analysis
maxrul_u = train.groupby('unit_number').max().reset_index()
maxrul_u.head()


# Scaling dataset
# 

# In[7]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# drop_labels = index_names+setting_names
drop_labels = index_names + ['RUL'] + setting_names
remain_labels = sensor_names

train_unit = train[train['unit_number']==1]
X_train=train.drop(columns=drop_labels).copy()
y_train = train['RUL'].copy()


X_train_scal=scaler.fit_transform(X_train)
X_train_scal_df = pd.DataFrame(X_train_scal, columns=remain_labels)


# Drop constant sensors

# In[8]:


# Check constant sensors
print(X_train_scal_df.iloc[:, :].describe().transpose())
const_labels = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']
tmp_labels = drop_labels + const_labels
remain_labels = [n for n in col_names if n not in tmp_labels]
# print(remain_labels)

X_train_scal_df = X_train_scal_df.drop(columns=const_labels)
X_train_scal = X_train_scal_df.values


# Smooth dataset

# In[9]:


from scipy.signal import savgol_filter
from mysindy_util import generate_list_increasing_by_dt

window_length = 11
poly_order = 2

X_train_scal_smoot_df = X_train_scal_df.apply(lambda col: savgol_filter(col, window_length, poly_order))
X_train_scal_smoot = X_train_scal_smoot_df.values

# Xdot
differentiation_method = ps.FiniteDifference(order=2)

# differentiation_method._differentiate(X, t)
t_max = X_train_scal_smoot.shape[0]
t = generate_list_increasing_by_dt(size=t_max, dt=1)
Xdot = differentiation_method._differentiate(X_train_scal_smoot, t)

print(X_train_scal_smoot.shape)


# In[10]:


training_data = {
    't': t,
    'x': X_train_scal_smoot,
    'dx': Xdot
}
validation_data = training_data


# ### SINDy

# In[11]:


params = {}

params['input_dim'] = len(remain_labels)
params['latent_dim'] = 10
params['model_order'] = 1
params['poly_order'] = 3
params['include_sine'] = True
params['library_dim'] = library_size(params['latent_dim'], params['poly_order'], params['include_sine'], True)

# sequential thresholding parameters
params['sequential_thresholding'] = True
params['coefficient_threshold'] = 0.05
params['threshold_frequency'] = 500
params['coefficient_mask'] = np.ones((params['library_dim'], params['latent_dim']))
params['coefficient_initialization'] = 'constant'

# loss function weighting
params['loss_weight_decoder'] = 1.0
params['loss_weight_sindy_z'] = 0.0
params['loss_weight_sindy_x'] = 1e-4
params['loss_weight_sindy_regularization'] = 1e-5

params['activation'] = 'sigmoid'
params['widths'] = [64,32]

# training parameters
params['epoch_size'] = training_data['x'].shape[0]
params['batch_size'] = 128
params['learning_rate'] = 1e-3

params['data_path'] = os.getcwd() + '/'
params['print_progress'] = True
params['print_frequency'] = 100
params['save_progress'] = True
params['save_frequency'] = 200

# training time cutoffs
params['max_epochs'] = 5001
params['refinement_epochs'] = 1001


# In[12]:


import datetime

num_experiments = 1
df = pd.DataFrame()
for i in range(num_experiments):
    print('EXPERIMENT %d' % i)

    params['coefficient_mask'] = np.ones((params['library_dim'], params['latent_dim']))

    params['save_name'] = './trained_models/cmapss_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

    tf.reset_default_graph()
    # tf.compat.v1.get_default_graph()

    results_dict = train_network(training_data, validation_data, params)
    df = df.append({**results_dict, **params}, ignore_index=True)

df.to_pickle('experiment_results_' + datetime.datetime.now().strftime("%Y%m%d%H%M") + '.pkl')


# In[ ]:




