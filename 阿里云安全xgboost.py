# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 11:52:42 2018

@author: Dell
"""

#Y = label.label
#X = data_train_simple.drop(['file_id'],axis = 1)

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve
from sklearn.model_selection import train_test_split


data_train_simple = pd.read_csv('E:/tianchi/阿里云安全/process_data/process_train.csv',header = 0)

Y = data_train_simple.label
X = data_train_simple.drop(['file_id','label'],axis =1)

'''
seed = 7
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=test_size, random_state=seed)

dTrain = xgb.DMatrix(X_train ,label=y_train)
dTest = xgb.DMatrix(X_test, label = y_test)

params={'booster':'gbtree',
    'objective': 'multi:softprob',
    'eval_metric':'mlogloss',
    'gamma':0.1,
    'num_class':8,
    'min_child_weight':0.8,
    'max_depth':6,
    'lambda':10,
    'subsample':0.7,
    'colsample_bytree':0.7,
    'colsample_bylevel':0.7,
    'eta': 0.02,
    'tree_method':'exact',
    'seed':0,
   'nthread':12
    }
watchlist = [(dTrain,'train'),(dTest,'test')]

model = xgb.train(params,dTrain,num_boost_round=1500,evals=watchlist)

model.save_model('E:/tianchi/阿里云安全/process_data/xgbmodel4')
#model=xgb.Booster(params)
#model.load_model('E:/tianchi/ofo/processed_data/xgbmodel2') 
'''

###---------------------------------------------------------------------------

data = pd.read_csv('E:/tianchi/阿里云安全/security_train/security_train.csv',header = 0)
data.file_id = data.file_id.astype('int16')
data.label = data.label.astype('int8')
data.tid = data.tid.astype('int16')
data.drop(['index'],axis = 1,inplace = True)


api_data = data[['file_id','api']]
api_data['num_api_used'] = 1
api_data = api_data.groupby(['file_id','api']).agg('sum').reset_index()
#观察误差情况
#-------------------------------------------------------------------------
a= 100



data_train_simple.index = data_train_simple.file_id
data_train_simple = data_train_simple.iloc[:,1:]


###可以多打印两轮
for i in range(684125):
    if i/1000==0:
        print(i)
    file_id = api_data.iloc[i,0]
    api = api_data.iloc[i,1]
    api = 'api_' + api
    num = api_data.iloc[i,2]
    
    try:
        data_train_simple[api][file_id]=num
        
    except:
        continue;

data_train_simple.to_csv('E:/tianchi/阿里云安全/process_data/processed_data20190103.csv')

import numpy as np
###api出现的前后顺序
import pandas as pd
data = pd.read_csv('E:/tianchi/阿里云安全/process_data/process_train_and_Sequence.csv')



#3## 减少内存消耗的  从kaggle拿过来的
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

data = reduce_mem_usage(data)


#------------------------------api_sequence导入，把api_sequence的列取出来
import pandas as pd
import numpy as np
data = pd.read_csv('E:/tianchi/阿里云安全/process_data/process_train_and_Sequence.csv')




#API连的列
api_sequence = data.columns.tolist()[922:]
#导入原始数据
data_test = pd.read_csv('E:/tianchi/阿里云安全/security_test/security_test.csv')
#改格式
data_test.file_id = data_test.file_id.astype('int16')
#data_test.label = data_test.label.astype('int8')
data_test.tid = data_test.tid.astype('int16')
data_test['index'] = data_test['index'].astype('int16')


data_test.head()
#只要这三列
data_test = data_test[['file_id','api','index']]

data_test_simple = pd.read_csv('E:/tianchi/阿里云安全/process_data/data_test_simple.csv')
#将data—_test_simple也添加800列
for api in api_sequence:
    data_test_simple[api] = 0
    data_test_simple[api] = data_test_simple[api].astype('int16')


data_test_simple.index = data_test_simple.file_id
data_test_simple.drop(['file_id'],inplace = True,axis = 1)


###横着比较稳
n = 0
#sequence
for i in data_test_simple.index.tolist():
    if i%10 ==0:
        print('i:',i)
    api_sequence_dict = {}    
    for j in range(n,len(data_test)-1):
        
        if data_test.iloc[j,0]==i:
            if data_test.iloc[j+1,2] - data_test.iloc[j,2]==1:
                #两个API连起来
                api_lian = '{}_{}'.format(data_test.iloc[j,1],data_test.iloc[j+1,1])
                #API1_API2
                if api_lian in api_sequence:
                    #在那800个里面的话算重要的API连 程序
                    try:
                        #加一
                        num = api_sequence_dict[api_lian]+1
                        api_sequence_dict[api_lian] = num
                        
                    except:
                        #如果字典里面没有这个键值对就添加进去 值为1
                        api_sequence_dict[api_lian] = 1
                else:
                    continue;
                
            else:
                continue;
        else:
            n = j
            break;
            
    for key,value in api_sequence_dict.items():#记得item要加()
        try:
            #先列后行
            #将所得数据输入进data_train_simple
            data_test_simple[key][i] = value
        except:
            print('can not pass the value')
            continue;
            
data_test_simple.to_csv('E:/tianchi/阿里云安全/process_data/data_test_and_sequence.csv')           
            
#----------------处理测试机数据
            
data = pd.read_csv('E:/tianchi/阿里云安全/security_test/security_test.csv',header = 0)
data.file_id = data.file_id.astype('int16')
#data.label = data.label.astype('int8')
data.tid = data.tid.astype('int16')
data.drop(['index'],axis = 1,inplace = True)


api_data = data[['file_id','api']]
api_data['num_api_used'] = 1
api_data = api_data.groupby(['file_id','api']).agg('sum').reset_index()

data_test_simple = pd.read_csv('E:/tianchi/阿里云安全/process_data/data_test_simple.csv')
data_test_simple.index = data_test_simple.file_id
data_test_simple = data_test_simple.iloc[:,1:]








#----------------------
data_new = pd.read_csv('E:/tianchi/阿里云安全/processed_data20190117.csv')

#data_new = data_new.iloc[:,:922]
data_new.index = data_new.file_id

data_new = data_new.iloc[:,1:]

testing_data = pd.concat([data_new,data_test_simple.iloc[:,920:]],axis = 1)
testing_data.to_csv('E:/tianchi/阿里云安全/processed_data20190117.csv')


