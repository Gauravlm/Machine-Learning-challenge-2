'''
Hacker Earth:-  #Machine Learning Challenge-2
subimission.csv  : project_Id and final status
'''

import pandas as pd
import numpy as np
import seaborn as sns


# load the taining data
train_data= pd.read_csv('train.csv')
test_data= pd.read_csv('test.csv')

train_data.shape   # rows= 108129 col= 14

train_data.describe()
##find correaltion between variable
#print(train_data.corr())
#
#corr= train_data.corr()

train_data.info()

train_data['disable_communication'].value_counts()

train_data.apply(lambda x: sum(x.isnull()))

# convert time to unix format
import datetime

unix_tcol = ['deadline','state_changed_at','launched_at','created_at']

for x in unix_tcol:
    train_data[x] = train_data[x].apply(lambda k: datetime.datetime.fromtimestamp(int(k)).strftime('%Y-%m-%d %H:%M:%S'))
    test_data[x] = test_data[x].apply(lambda k: datetime.datetime.fromtimestamp(int(k)).strftime('%Y-%m-%d %H:%M:%S'))
    
 
#  add  some features    
col_to_use= ['name','desc']
len_feats= ['name_len','desc_len']
count_feats= ['name_counts','desc_counts']

for i in np.arange(2):
    train_data[i] = train_data[col_to_use[i]].str.split().str.len()  
    test_data[i]= test_data[col_to_use[i]].str.split().str.len()
    
    
train_data['name_counts']=train_data['name'].str.split().str.len()
train_data['desc_counts']=train_data['desc'].str.split().str.len()   

test_data['name_counts']=train_data['name'].str.split().str.len()
test_data['desc_counts']=train_data['desc'].str.split().str.len()   

train_data['keyword_len']= train_data['keywords'].str.len()
test_data['keyword_len']= test_data['keywords'].str.len()

train_data['keyword_count']= train_data['keywords'].str.split('-').str.len()
test_data['keyword_count']= test_data['keywords'].str.split('-').str.len()


# adding more imformation
for x in unix_tcol:
    train_data[x]=train_data[x].apply(lambda k: datetime.datetime.strptime(k,'%Y-%m-%d %H:%M:%S'))
    test_data[x]=test_data[x].apply(lambda k: datetime.datetime.strptime(k,'%Y-%m-%d %H:%M:%S'))

# here we are creating difference between 
# 1:- Launch_at and created_at
# 2:- deadling_at and launch_at

time1 =[]
time2 = []
for i in np.arange(train_data.shape[0]):
    time1.append(np.round((train_data.loc[i,'launched_at'] - train_data.loc[i,'created_at']).total_seconds()).astype('int'))
    time2.append(np.round((train_data.loc[i,'deadline'] - train_data.loc[i,'launched_at']).total_seconds()).astype('int'))

train_data['time1']= np.log(time1)
train_data['time2']= np.log(time2)
