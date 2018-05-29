import xlearn as xl
import pandas as pd
import numpy as np
import os
path='../data/'
print('reading test file...', flush=True)
# len_test = 200
test_df = pd.read_csv(path + 'test2.csv')  # , nrows=len_test
ffm_model = xl.create_ffm()
ffm_model.setTrain(path+'train_ffm.txt')
ffm_model.setValidate(path+'valid_ffm.txt')
ffm_model.setTest(path+'test_ffm.txt')
ffm_model.setSigmoid()
param = {   'task':'binary', 
			'lr':0.01, 
			'lambda':0.001,
			'metric': 'auc',
			'opt':'ftrl',
			'epoch':20,
			'k':4,
			'alpha': 1.5, 
			'beta': 0.01, 
			'lambda_1': 0.0, 
			'lambda_2': 0.0,
			'stop_window':10
		}
print('training...', flush=True)
ffm_model.fit(param,path + "ffm_model.out")
print('predicting...', flush=True)
ffm_model.predict(path + "ffm_model.out",path + "predict.txt")
sub = pd.DataFrame()
sub['aid']=test_df['aid']
sub['uid']=test_df['uid']
sub['score'] = np.loadtxt(path + "predict.txt")
print('writing results...', flush=True)
sub.to_csv('submission.csv',index=False)
os.system('zip baseline_ffm.zip submission.csv')
