from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import HashingVectorizer
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import lightgbm as lgb
import os
import gc
import multiprocessing
import time
import sys
from multiprocessing import Pool
from multiprocessing import cpu_count


##################################################
#          LightGBM model declaration            #
##################################################
def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
				 feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, categorical_features=None):
	lgb_params = {
		'boosting_type': 'gbdt',
		'objective': objective,
		'metric': metrics,
		'learning_rate': 0.01,
		# 'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
		'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
		'max_depth': -1,  # -1 means no limit
		'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
		'max_bin': 255,  # Number of bucketed bin for feature values
		'subsample': 0.6,  # Subsample ratio of the training instance.
		'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
		'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
		'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
		'subsample_for_bin': 200000,  # Number of samples for constructing bin
		'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
		'reg_alpha': 0,  # L1 regularization term on weights
		'reg_lambda': 0,  # L2 regularization term on weights
		'nthread': 8,
		'verbose': 0,
	}

	lgb_params.update(params)

	print("preparing validation datasets...")

	xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
						  feature_name=predictors,
						  categorical_feature=categorical_features
						  )
	xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
						  feature_name=predictors,
						  categorical_feature=categorical_features
						  )

	evals_results = {}

	bst1 = lgb.train(lgb_params,
					 xgtrain,
					 valid_sets=[xgtrain, xgvalid],
					 valid_names=['train', 'valid'],
					 evals_result=evals_results,
					 num_boost_round=num_boost_round,
					 early_stopping_rounds=early_stopping_rounds,
					 verbose_eval=10,
					 feval=feval)

	n_estimators = bst1.best_iteration
	print("\nModel Report")
	print("n_estimators : ", n_estimators)
	print(metrics+":", evals_results['valid'][metrics][n_estimators-1])

	return bst1


##################################################
#          vector feature pre-process            #
##################################################
# Method 1: using word2vec model
def base_word2vec(x, model, size):
	vec = np.zeros(size)
	x = [item for item in x if model.wv.__contains__(item)]
	for item in x:
		vec += model.wv[item]
	if len(x) == 0:
		return vec
	else:
		return vec / len(x)


def w2v_feature_process(feature):
	global data
	print('processing', feature)
	data[feature] = data[feature].apply(lambda x: str(x).split(' '))
	model = Word2Vec(data[feature], size=10, min_count=1, iter=5, window=2)
	data_vec = []
	for row in data[feature]:
		data_vec.append(base_word2vec(row, model, size=10))
	column_names = []
	for i in range(10):
		column_names.append(feature + str(i))
	data_vec = pd.DataFrame(data_vec, columns=column_names)
	return data_vec


##################################################
#          one-hot feature pre-process           #
##################################################
# Method 1: using OneHotEncoder
def one_hot_feature_process(feature, ohe):
	global data
	print('processing', feature)
	try:
		data[feature] = data[feature].apply(lambda x: str(x).split(' '))
		curr_col = ohe.fit_transform(data[feature].apply(int))
	except:
		curr_col = ohe.fit_transform(data[feature])
	curr_col = pd.DataFrame(curr_col)
	curr_col.columns = [feature]
	return curr_col


##################################################
#          feature pre-process                   #
##################################################
# processing vector & one-hot features
def feature_pre_process(one_hot_features, vector_features):
	global data
	n = 4
	new_features = []

	print('processing one-hot features...')
	p = Pool(n)
	# ohe = OneHotEncoder()
	ohe = LabelEncoder()
	for feature in one_hot_features:
		new_features.append(p.apply_async(one_hot_feature_process, args=(feature, ohe,)))
	p.close()
	p.join()

	for new_feature in new_features:
		data = pd.concat([data, new_feature.get()], axis=1)
	new_features.clear()
	gc.collect()
	print('one-hot done!')

	print('processing w2v features...')
	q = Pool(n)
	for feature in vector_features:
		new_features.append(q.apply_async(w2v_feature_process, args=(feature,)))
	q.close()
	q.join()

	for new_feature in new_features:
		data = pd.concat([data, new_feature.get()], axis=1)
	del new_features
	gc.collect()
	print('w2v done!')



#######################################################################
#          feature engineering with multiprocessing                   #
#######################################################################
# split data into chunks
def get_chunks(group,n_split):
	groupname =  list(group.count().index.values)
	chunksname = []
	totallength = len(groupname)
	chunksize = int(totallength/n_split)
	for i in range(n_split-1):
		chunksname.append(groupname[i*chunksize:(i+1)*chunksize])
	chunksname.append(groupname[(n_split-1)*chunksize:])
	group_chunks = []
	for i in range(n_split):
		group_chunks.append([])
		for user in chunksname[i]:
			group_chunks[-1].append(group.get_group(user))
	
	return group_chunks
# feature engineering for each chunk
def chunks_processing(chunk,i):
	total = 0
	for id_group in chunk:
		total += id_group['uid'].value_counts().count()
	num = 0
	id_groups = []
	start = time.time()
	for id_group in chunk:
		id_group = id_group.reset_index(drop=True)
		# do stat feature engineering here
		age_count = id_group['age'].value_counts()
		for index,features in id_group.iterrows():
			# do feature engineering by row here
			if(id_group['appIdInstall'][index] == '-1'):
				id_group['appIdAction'][index] = '-1'   #for example

		id_groups.append(id_group)
		num += 1
		end = time.time()
		usetime = end - start
		#view_bar(num, total,usetime,i)
	chunk = pd.concat(id_groups)
	chunk = chunk.reset_index(drop=True)

	return chunk
# feature engineering with multiprocessing
def feature_engineering_mp(data):
	n_cores = 4
	uid_groups = []
	for uid,uid_group in data.groupby('uid'):
		uid_groups.append(uid_group)
	data = pd.concat(uid_groups).reset_index(drop=True)
	data_byid = data.groupby('uid')
	chunks = get_chunks(data_byid,n_cores)

	pool = multiprocessing.Pool(n_cores)
	chunks = pool.starmap(chunks_processing,zip(chunks,range(n_cores)))

	data = pd.concat(chunks).reset_index(drop=True)

	return data

##################################################
#          main function                         #
##################################################
if __name__ == '__main__':
	print('############################################################\n')
	print('           Tencent Algo Competition    by HBZ               \n')
	print('############################################################\n')

	one_hot_features = ['marriageStatus', 'ct', 'os']
	vector_features = ['interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1',
					   'topic2', 'topic3', 'appIdInstall', 'appIdAction']
	categorical_features = ['productType', 'age', 'gender', 'education', 'consumptionAbility', 'carrier', 'house']
	other_features = ['aid', 'advertiserId', 'campaignId', 'creativeSize', 'adCategoryId', 'productId', 'uid', 'LBS']
	features = ['aid', 'label', 'uid', 'advertiserId', 'campaignId', 'creativeSize', 'adCategoryId', 'productId',
				'productType', 'age', 'gender', 'marriageStatus', 'education', 'consumptionAbility', 'LBS', 'interest1',
				'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3',
				'appIdInstall', 'appIdAction', 'ct', 'os', 'carrier', 'house', 'marriageStatus.1', 'ct.1', 'os.1',
				'interest10', 'interest11', 'interest12', 'interest13', 'interest14', 'interest15', 'interest16',
				'interest17', 'interest18', 'interest19', 'interest20', 'interest21', 'interest22', 'interest23',
				'interest24', 'interest25', 'interest26', 'interest27', 'interest28', 'interest29', 'interest30',
				'interest31', 'interest32', 'interest33', 'interest34', 'interest35', 'interest36', 'interest37',
				'interest38', 'interest39', 'interest40', 'interest41', 'interest42', 'interest43', 'interest44',
				'interest45', 'interest46', 'interest47', 'interest48', 'interest49', 'interest50', 'interest51',
				'interest52', 'interest53', 'interest54', 'interest55', 'interest56', 'interest57', 'interest58',
				'interest59', 'kw10', 'kw11', 'kw12', 'kw13', 'kw14', 'kw15', 'kw16', 'kw17', 'kw18', 'kw19', 'kw20',
				'kw21', 'kw22', 'kw23', 'kw24', 'kw25', 'kw26', 'kw27', 'kw28', 'kw29', 'kw30', 'kw31', 'kw32', 'kw33',
				'kw34', 'kw35', 'kw36', 'kw37', 'kw38', 'kw39', 'topic10', 'topic11', 'topic12', 'topic13', 'topic14',
				'topic15', 'topic16', 'topic17', 'topic18', 'topic19', 'topic20', 'topic21', 'topic22', 'topic23',
				'topic24', 'topic25', 'topic26', 'topic27', 'topic28', 'topic29', 'topic30', 'topic31', 'topic32',
				'topic33', 'topic34', 'topic35', 'topic36', 'topic37', 'topic38', 'topic39', 'appIdInstall0',
				'appIdInstall1', 'appIdInstall2', 'appIdInstall3', 'appIdInstall4', 'appIdInstall5', 'appIdInstall6',
				'appIdInstall7', 'appIdInstall8', 'appIdInstall9', 'appIdAction0', 'appIdAction1', 'appIdAction2',
				'appIdAction3', 'appIdAction4', 'appIdAction5', 'appIdAction6', 'appIdAction7', 'appIdAction8', 'appIdAction9']

	path = '../preliminary_contest_data/'  # set file path
	len_train = 8798814  # training set size
	len_test = 2265989  # test set size
	len_valid = 1700000  # set size of cross-validation set


##################################################
#          loading original data                 #
##################################################
	# extracting features from original datasets
	# if not os.path.exists(path + 'processed_data_sample.csv'):
	'''
		print('loading train...')
		data = pd.read_csv(path + "train.csv")
		print('loading test...')
		test = pd.read_csv(path + "test1.csv")
		print('loading adFeature...')
		adF = pd.read_csv(path + "adFeature.csv", usecols=['aid', 'advertiserId', 'campaignId', 'creativeSize', 'adCategoryId', 'productId', 'productType'])
		print('loading userFeature...')
		userF = pd.read_csv(path + "userFeature.csv")

		print('---------------data pre-processing---------------')
		len_train = len(data)
		data = data.append(test)
		del test
		gc.collect()

		print('merging...')
		data = data.merge(adF, on='aid', how='left')
		del adF
		gc.collect()

		data = data.merge(userF, on='uid', how='left')
		del userF
		gc.collect()
	'''
	data = pd.read_csv(path + "data_sample.csv")
	data.fillna('-1', inplace=True)
	print('extracting...')
	feature_pre_process(one_hot_features, vector_features)
	print('writing processed_data.csv...')
	data.to_csv(path + 'processed_data_sample.csv', index=True)
	'''
	else:
		print('loading existing data...')
		use = set(features)
		use = list(use - use.intersection(set(one_hot_features)) - use.intersection(set(vector_features)))
		data = pd.read_csv(path + 'processed_data_sample.csv', usecols=use)
	'''


##################################################
#          feature engineering                   #
##################################################
	print('adding count feature of ids...')
	id_features = ['aid', 'advertiserId', 'campaignId', 'adCategoryId', 'productId', 'uid']
	for id_feature in id_features:
		idCount = data[id_feature].value_counts()
		idCount = pd.DataFrame({id_feature: idCount.index, id_feature + 'Count': idCount.values})
		data = data.merge(idCount, on=id_feature, how='left')
	print("---------------multiprocessing features---------")
	feature_engineering_mp(data)

	print("---------------all features done!---------------")


##################################################
#          data info & training preparation      #
##################################################
	print("vars and data type: ")
	data.info()
	pd.set_option('display.max_colwidth', 1000)
	print(data.head(5))

	# splitting train/valid/test
	train = data[:(len_train - len_valid)]
	valid = data[(len_train - len_valid):len_train]
	test = data[len_train:]
	del data
	gc.collect()
	print("train size: ", len(train))
	print("valid size: ", len(valid))
	print("test size : ", len(test))
	print("---------------ready to run!---------------\n")

	target = 'label'
	predictors = train.columns.values.tolist()
	predictors.remove('label')
	categorical = categorical_features

	submission = pd.DataFrame()
	submission['aid'] = test['aid']
	submission['uid'] = test['uid']
	gc.collect()


##################################################
#          model params setting & running        #
##################################################
	print("---------------training...---------------")
	'''
	params = {
		'learning_rate': 0.05,
		# 'is_unbalance': 'true', # replaced with scale_pos_weight argument
		'num_leaves': 15,  # we should let it be smaller than 2^(max_depth)
		'max_depth': 4,  # -1 means no limit
		'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
		'max_bin': 100,  # Number of bucketed bin for feature values
		'subsample': 0.7,  # Subsample ratio of the training instance.
		'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
		'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
		'min_child_weight': 50,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
		'reg_lambda': 1,  # L2 regularization term on weights
		'scale_pos_weight': 95  # because training data is extremely unbalanced
	}
	bst = lgb_modelfit_nocv(params,
							train,
							valid,
							predictors,
							target,
							objective='binary',
							metrics='auc',
							early_stopping_rounds=50,
							verbose_eval=True,
							num_boost_round=2000,
							categorical_features=categorical)

	del train
	del valid
	gc.collect()

	print("---------------predicting...---------------")
	submission['score'] = bst.predict(test[predictors])
	print("---------------writing...---------------")
	submission.to_csv('submission.csv', index=False)
	print("---------------all done!---------------")
	print(submission.info())
	'''
