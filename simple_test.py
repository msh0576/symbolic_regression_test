

import numpy as np
import os
import pyreadr
import matplotlib.pyplot as plt

dataset_path = './Dataset/TE_process'
file_name = 'd00_te.dat'


file_path = os.path.join(dataset_path, file_name)


X = np.loadtxt(file_path)    # (960, 52)


train_normal_path = './Dataset/TEP_FaultFree_Training.RData'
train_faulty_path = './Dataset/TEP_Faulty_Training.RData'


test_normal_path = './Dataset/TEP_FaultFree_Testing.RData'
test_faulty_path = './Dataset/TEP_Faulty_Testing.RData'


train_normal_complete = pyreadr.read_r(train_normal_path)['fault_free_training']
#train_faulty_complete = pyreadr.read_r(train_fault_path)['faulty_training']

#test_normal_complete = pyreadr.read_r(test_normal_path)['fault_free_testing']
test_faulty_complete = pyreadr.read_r(test_faulty_path)['faulty_testing']

df_train = train_normal_complete[train_normal_complete.simulationRun==1].iloc[:,3:]

fig, ax = plt.subplots(13,4,figsize=(30,50))

for i in range(df_train.shape[1]):
    df_train.iloc[:,i].plot(ax=ax.ravel()[i]) 
    ax.ravel()[i].legend();
# plt.show()