import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
from sklearn.grid_search import GridSearchCV
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import scipy
from tqdm import tqdm
import gc
import pickle

print("reading..")
ad_feature=pd.read_csv('../data/adFeature.csv')
if os.path.exists('../data/userFeature.csv'):
    user_feature=pd.read_csv('../data/userFeature.csv')
else:
    userFeature_data = []
    with open('../data/userFeature.data', 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split('|')
            userFeature_dict = {}
            for each in line:
                each_list = each.split(' ')
                userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
            userFeature_data.append(userFeature_dict)
            if i % 100000 == 0:
                print(i)
        user_feature = pd.DataFrame(userFeature_data)
        user_feature.to_csv('../data/userFeature.csv', index=False)
train=pd.read_csv('../data/train.csv')
predict=pd.read_csv('../data/test1.csv')
print("read prepared!")
train.loc[train['label']==-1,'label']=0
predict['label']=-1
data=pd.concat([train,predict])
data=pd.merge(data,ad_feature,on='aid',how='left')
data=pd.merge(data,user_feature,on='uid',how='left')
data=data.fillna('-1')

names = []
group_features = ['aidCount', 'uidCount', 'productType', 'age', 'gender', 'education', 'consumptionAbility',
                  'appIdInstall.1', 'carrier', 'house', 'marriageStatus.1', 'ct.1', 'os.1']
for i in range(len(group_features) - 1):
    for j in range(i + 1, len(group_features)):
        name = group_features[i] + '_' + group_features[j]
        names.append(name)
group_user_features = ['age', 'gender', 'education', 'consumptionAbility', 'house', 'marriageStatus.1', 'os.1']
for i in range(len(group_user_features) - 2):
    for j in range(i + 1, len(group_user_features) - 1):
        for k in range(j + 1, len(group_user_features)):
            name = group_user_features[i] + '_' + group_user_features[j] + '_' + group_user_features[k]
            names.append(name)



path = '../data/' 
fe = pd.read_csv(path + 'engineered_features.csv', usecols=names)
data = pd.concat([data, fe], axis = 1)

del fe
gc.collect()

fe_2 = pd.read_csv(path + 'engineered_features_2.csv')
data = pd.concat([data, fe_2], axis = 1)

del fe_2
gc.collect()

fe_ff = pd.read_csv(path + 'fe_ff.csv')
data = pd.concat([data, fe_ff], axis = 1)

print("Concat Complete")


one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','advertiserId','campaignId', 'creativeId',
       'adCategoryId', 'productId', 'productType']
vector_feature=['marriageStatus','appIdAction','appIdInstall','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']
for feature in one_hot_feature:
    try:
        data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
    except:
        data[feature] = LabelEncoder().fit_transform(data[feature])

train=data[data.label!=-1]
train_y=train.pop('label')
train, evals, train_y, evals_y = train_test_split(train,train_y,test_size=0.2, random_state=2018)
test=data[data.label==-1]
res=test[['aid','uid']]
test=test.drop('label',axis=1)

ff_f_names = ['interest1_length',
 	'interest2_length',
	 'interest3_length',
	 'interest4_length',
	 'interest5_length',
	 'kw1_length',
	 'kw2_length',
	 'kw3_length',
	 'topic1_length',
	 'topic2_length',
	 'topic3_length',
	 'vector_num',
	 'map_advertiserId_campaignId',
	 'map_campaignId_aid']


bb_f_2_names = ['aid_ct.1',
	 'aid_campaignId',
	 'aid_advertiserId',
	 'aid_marriageStatus.1',
	 'aid_age',
	 'ct.1_campaignId',
	 'ct.1_advertiserId',
	 'ct.1_marriageStatus.1',
	 'ct.1_age',
	 'campaignId_advertiserId',
	 'campaignId_marriageStatus.1',
	 'campaignId_age',
	 'advertiserId_marriageStatus.1',
	 'advertiserId_age',
	 'marriageStatus.1_age',
	 'aid_ct.1_campaignId',
	 'aid_ct.1_advertiserId',
	 'aid_ct.1_marriageStatus.1',
	 'aid_ct.1_age',
	 'aid_campaignId_advertiserId',
	 'aid_campaignId_marriageStatus.1',
	 'aid_campaignId_age',
	 'aid_advertiserId_marriageStatus.1',
	 'aid_advertiserId_age',
	 'aid_marriageStatus.1_age',
	 'ct.1_campaignId_advertiserId',
	 'ct.1_campaignId_marriageStatus.1',
	 'ct.1_campaignId_age',
	 'ct.1_advertiserId_marriageStatus.1',
	 'ct.1_advertiserId_age',
	 'ct.1_marriageStatus.1_age',
	 'campaignId_advertiserId_marriageStatus.1',
	 'campaignId_advertiserId_age',
	 'campaignId_marriageStatus.1_age',
	 'advertiserId_marriageStatus.1_age']

cols = names + ['creativeSize', 'aid'] + ff_f_names + bb_f_2_names

train_x = train[cols]
evals_x = evals[cols]
test_x = test[cols]

enc = OneHotEncoder()
for feature in one_hot_feature:
    enc.fit(data[feature].values.reshape(-1, 1))
    train_a=enc.transform(train[feature].values.reshape(-1, 1))
    test_a = enc.transform(test[feature].values.reshape(-1, 1))
    evals_a=enc.transform(evals[feature].values.reshape(-1, 1))
    train_x= sparse.hstack((train_x, train_a))
    evals_x = sparse.hstack((evals_x, evals_a))
    test_x = sparse.hstack((test_x, test_a))
print('one-hot prepared !')


cv=CountVectorizer()
for i in tqdm(range(len(vector_feature))):
    feature = vector_feature[i]
    cv.fit(data[feature])
    train_a = cv.transform(train[feature])
    test_a = cv.transform(test[feature])
    evals_a=cv.transform(evals[feature])
    train_x = sparse.hstack((train_x, train_a))
    test_x = sparse.hstack((test_x, test_a))
    evals_x=sparse.hstack((evals_x,evals_a))
print('cv prepared !')
print('shape of data: ')
print(data.shape)

del data
gc.collect()

print("LGB test")
clf = lgb.LGBMClassifier(
    boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=40000, objective='binary',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.02, min_child_weight=50, random_state=2018, n_jobs=-1
)
clf.fit(train_x, train_y, eval_set=[(evals_x, evals_y)], eval_metric='auc',early_stopping_rounds=2000)


res['score'] = clf.predict_proba(test_x)[:,1]
res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
res.to_csv('../submission_bb_ff_0.2_40000.csv', index=False)

with open('model_bb_ff_0.2_40000.pkl', 'wb') as f:
	pickle.dump(clf, f)


