import pandas as pd
import numpy as np
import os
import gc

from sklearn.model_selection import train_test_split, KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

def read_data(file, len_train):
	data = pd.read_csv(file)
	data = data.fillna("-1")
	train = data[:len_train]
	test = data[len_train:]
	x = np.array(train.drop(['label'],axis=1))
	y = np.array(train['label'])
	x_train,x_valid,y_train,y_valid = train_test_split(x, y,test_size=0.2,random_state=2018)

	x_test = np.array(test.drop(['label'],axis=1))

	del data
	del train
	gc.collect()

	return x_train, y_train, x_valid, y_valid, x_test, test

def stacking(X, Y, Tx, Ty, x_test):
	random_rate = 2018

	# classifiers in layer 1
	clf1 = lgb.LGBMClassifier(
		boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
		max_depth=-1, n_estimators=40000, objective='binary',
		subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
		learning_rate=0.02, min_child_weight=50, random_state=2018, n_jobs=-1
	)
	clf2 = lgb.LGBMClassifier(
		boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
		max_depth=-1, n_estimators=40000, objective='binary',
		subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
		learning_rate=0.02, min_child_weight=50, random_state=8102, n_jobs=-1
	)

	clfs = [
		['lgb1', clf1],
		['lgb2', clf2]

	]

	# classifier in layer 2
	clf_l2 = lgb.LGBMClassifier(
		boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
		max_depth=-1, n_estimators=40000, objective='binary',
		subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
		learning_rate=0.02, min_child_weight=50, random_state=9999, n_jobs=-1
	)

	S_train = np.zeros((X.shape[0], len(clfs)))
	S_valid = np.zeros((Tx.shape[0], len(clfs)))
	S_test = np.zeros((x_test.shape[0], len(clfs)))
	folds = list(StratifiedKFold(Y, n_folds=5, random_state=0))

	# layer 1 stacking
	for i, bm in enumerate(clfs):
		clf = bm[1]
		S_valid_i = np.zeros((Tx.shape[0], len(folds)))
		S_test_i = np.zeros((x_test.shape[0], len(folds)))
		S_valid_auc = []
		for j, (train_idx, test_idx) in enumerate(folds):

			X_train = X[train_idx]
			y_train = Y[train_idx]
			X_holdout = X[test_idx]
			y_holdout = Y[test_idx]

			clf.fit(X_train, y_train)
			y_pred = clf.predict(X_holdout)[:]
			S_train[test_idx, i] = y_pred
			S_valid_i[:, j] = clf.predict(Tx)[:]
			S_test_i[:, j] = clf.predict(x_test)[:]
			S_valid_auc.append(roc_auc_score(Ty, S_valid_i[:, j], average='macro'))
			# S_valid_Fscore.append(fbeta_score(Ty, S_valid_i[:, j], average='binary', beta=0.1))

		print("%s:\tTrain auc_score:%.4f\tValid auc_score:%.4f"%(bm[0],roc_auc_score(Y, S_train[:, i], average='macro'),sum(S_valid_auc)/len(S_valid_auc)))
		
		S_valid[:, i] = S_valid_i.mean(1)
		S_test[:, i] = S_test_i.mean(1)

	# layer 2 stacking
	clf_l2.fit(S_train, Y)
	y_train_pred = clf_l2.predict(S_train)[:]
	y_valid_pred = clf_l2.predict(S_valid)[:]
	y_test_pred = clf_l2.predict(S_test)[:]
	print("Stack:\tTrain Fbeta_score:%.4f\tValid Fbeta_score:%.4f"%(roc_auc_score(Y, y_train_pred, average='macro'),roc_auc_score(Ty, y_valid_pred, average='macro')))
	return y_test_pred
	

if __name__ == '__main__':
	x_train, y_train, x_valid, y_valid, x_test, test_df = read_data('data.csv', 100) # specify the data path and len_train here
	pred = stacking(x_train, y_train, x_valid, y_valid, x_test)
	sub = pd.DataFrame()
	sub['aid'] = test_df['aid']
	sub['uid'] = test_df['uid']
	sub['score'] = pred
	del test_df
	gc.collect()
	print('writing results...')
	sub.to_csv('submission.csv',index=False)
	os.system('zip baseline_ffm.zip submission.csv')