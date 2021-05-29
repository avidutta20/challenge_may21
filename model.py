#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os, sys
from collections import defaultdict
# %%
# get_ipython().system('cd D:\\')

# In[2]:


# get_ipython().system('git clone https://github.com/avidutta20/challenge_may21')


# %%
# working through windows
# set the current working directory for based on the system
wdir_path = 'D:\\development\\git\\jobathon_may21\\'

# setting cwd to wdir_path3
os.chdir(wdir_path)

# In[3] %%

def load_data():
    pass
df = pd.read_csv('dataset\\av_jobathon_may21_train.csv')

# creating a list of all trainable features name
feature_list = df.columns[1:10]
target = df.columns[10]  # 'Is_Lead' column name

# splitting into train & test
x_train, x_test, y_train, y_test = train_test_split(df[feature_list], df[target], test_size=0.25, random_state=25)
    



# # %%
# # loading data
# df = pd.read_csv('dataset\\av_jobathon_may21_train.csv')

# # creating a list of all trainable features name
# feature_list = df.columns[1:10]
# target = df.columns[10]  # 'Is_Lead' column name

# # splitting into train & test
# x_train, x_test, y_train, y_test = train_test_split(df[feature_list], df[target], test_size=0.25, random_state=25)


#%%
def execute_process(dframe, colname,func,arg1):
    dframe[colname] = dframe[colname].apply(lambda x: func(x,arg1))
    return dframe
def extract_process(value,index):
    value = int(value[index:])
    return value

def replace_values(value, value_dic):
    # replaces any value in the dframe colname with given dict
    return value_dic[value]

# In[10]:
def default_value():
    return 0
# %%
gender_dic = is_active_dic = occupation_dic = defaultdict(default_value)
#%%
gender_dic = {'Male': 0,
                'Female': 1}
is_active_dic = {'Yes': 1,
                 'No': 0}

occupation_dic = {'Self_Employed': 1,
                     'Other': 2,
                     'Salaried': 3,
                     'Entrepreneur': 4}

execute_process(x_train,'Region_Code', extract_process, 2)
print('Region Code Processed Successfully')

execute_process(x_train,'Channel_Code', extract_process, 1)
print('Channel Code Processed Successfully')

execute_process(x_train,'Gender', replace_values, gender_dic)
print('Gender Processed Successfully')

execute_process(x_train,'Occupation', replace_values, occupation_dic)
print('Occupation Processed Successfully')

execute_process(x_train,'Is_Active',replace_values, is_active_dic)
print('Is Active Processed Successfully')

x_train['Credit_Product'] = x_train['Credit_Product'].fillna('No')
execute_process(x_train,'Credit_Product',replace_values, is_active_dic)


# %% Implementing Random Forests

clf_rforest = RandomForestClassifier()
clf_rforest.fit(x_train, y_train)

# %%
# getting important features
imp_feature = {}
for feat_name, weight in zip(feature_list, clf_rforest.feature_importances_):
    imp_feature[feat_name] = weight


for key in imp_feature:
    print(f'{key} : {round(imp_feature[key],5)*100}')

