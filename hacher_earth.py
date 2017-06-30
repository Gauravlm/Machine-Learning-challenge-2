'''
Hacker Earth:-  #Machine Learning Challenge-2
subimission.csv  : project_Id and final status
'''

import pandas as pd
import numpy as np
import seaborn as sns
from bs4 import BeautifulSoupb
import xgboost as xgb
# load taining data
train_data= pd.read_csv('train.csv')
# load test data
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
    train_data[len_feats[i]] = train_data[col_to_use[i]].apply(str).apply(len)  
    test_data[len_feats[i]]= test_data[col_to_use[i]].apply(str).apply(len)
    
    
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

# for test data
time3 =[]
time4 = []
for i in np.arange(test_data.shape[0]):
    time3.append(np.round((test_data.loc[i,'launched_at'] - test_data.loc[i,'created_at']).total_seconds()).astype('int'))
    time4.append(np.round((test_data.loc[i,'deadline'] - test_data.loc[i,'launched_at']).total_seconds()).astype('int'))

test_data['time1']= np.log(time3)
test_data['time2']= np.log(time4)

feat= ['disable_communication','country']
from sklearn.preprocessing import LabelEncoder

for i in feat:
    le= LabelEncoder()
    le.fit(list(train_data[i].values)+ list(test_data[i].values))
    train_data[i]= le.transform(list(train_data[i]))
    test_data[i]= le.transform(list(test_data[i]))

train_data['goal']= np.log1p(train_data['goal'])
test_data['goal']= np.log1p(test_data['goal'])


# cleaning Text 
#creating full list of descriptions from train and test data
tdesc = pd.Series(train_data['desc'].tolist()  + test_data['desc'].tolist()).astype(str)
#kdesc = pd.Series(train_data['desc'].tolist()  + test_data['desc'].tolist()).astype('str')
##########################################################################################
import re
def desc_clean(word):
    p1 = re.sub(pattern='(\W+)|(\d+)|(\s+)',repl=' ',string=word)
    p1 = p1.lower()
    return p1
'''
\W :	Matches nonword characters
\d:    Matches digits. Equivalent to [0-9].
\s:    Matches whitespace. Equivalent to [\t\n\r\f].                              
'''
tdesc= tdesc.map(desc_clean)

# or  

#def text_clean(raw):
#    letters_only = re.sub("[^a-zA-Z\s]", " ", raw)
#    letters_only= re.sub("(\s+)", " ", letters_only)
#    return letters_only.lower()
#
#kdesc= kdesc.map(text_clean)

from nltk.corpus import stopwords

stop_word= set(stopwords.words('english'))
tdesc = [[x for x in x.split() if x not in stop_word] for x in tdesc ]


from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer(language= 'english')
tdesc = [[stemmer.stem(x) for x in x] for x in tdesc]
tdesc = [[x for x in x if len(x) > 2] for x in tdesc]
tdesc = [' '.join(x) for x in tdesc]



# creating count features
from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features=650)
alldesc = cv.fit_transform(tdesc).todense()

alldesc_df = pd.DataFrame(alldesc)
alldesc_df.rename(columns = lambda x : 'varialbe_' + str(x),inplace= True)


# text features split
train_text= alldesc_df[:train_data.shape[0]]
test_text = alldesc_df[train_data.shape[0]:]

test_text.reset_index(drop= True, inplace= True)

cols_to_use = ['name_len','desc_len','keywords_len','name_count','desc_count','keywords_count','time1','time3','goal']
target=train_data['final_status']

train_data=train_data.loc[:,cols_to_use]
test_data= test_data.loc[:,cols_to_use]

x_train= pd.concat([train_data,train_text],axis=1)
x_test= pd.concat([test_data,test_text],axis=1)

print(x_train.shape)
print(x_test.shape)




# Model Training 
dtrain= xgb.DMatrix(data=x_train,label= target)
dtest= xgb.DMatrix(data=x_test)

params = {
    'objective':'binary:logistic',
    'eval_metric':'error',
    'eta':0.025,
    'max_depth':6,
    'subsample':0.7,
    'colsample_bytree':0.7,
    'min_child_weight':5}
    
bst = xgb.cv(params, dtrain, num_boost_round=1000, early_stopping_rounds=40, nfold=5, verbose_eval=10)

bst_train = xgb.train(params, dtrain, num_boost_round=1000)
p_test = bst_train.predict(dtest)

sub = pd.DataFrame()
sub['project_id'] = test_data['project_id']
sub['final_status'] = p_test
   
sub['final_status'] = [1 if x > 0.5 else 0 for x in sub['final_status']]
sub.to_csv("xgb_with_python_feats.csv",index=False)   