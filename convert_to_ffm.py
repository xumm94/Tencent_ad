import pandas as pd
import numpy as np
import os
import gc

path = '../data/'


interest1_feature = [	'interest1_0', 'interest1_1', 'interest1_2', 'interest1_3', 'interest1_4',
                      	'interest1_5', 'interest1_6', 'interest1_7', 'interest1_8', 'interest1_9']
interest2_feature = [	'interest2_0', 'interest2_1', 'interest2_2', 'interest2_3',
						'interest2_4', 'interest2_5', 'interest2_6', 'interest2_7',
						'interest2_8', 'interest2_9']
interest3_feature = [	'interest3_0', 'interest3_1',
						'interest3_2', 'interest3_3', 'interest3_4', 'interest3_5',
						'interest3_6', 'interest3_7', 'interest3_8', 'interest3_9']
interest4_feature = [	'interest4_0', 'interest4_1', 'interest4_2', 'interest4_3',
						'interest4_4', 'interest4_5', 'interest4_6', 'interest4_7',
						'interest4_8', 'interest4_9']
interest5_feature = [	'interest5_0', 'interest5_1',
						'interest5_2', 'interest5_3', 'interest5_4', 'interest5_5',
						'interest5_6', 'interest5_7', 'interest5_8', 'interest5_9']
kw1_feature = [			'kw1_0', 'kw1_1', 'kw1_2', 'kw1_3', 'kw1_4', 'kw1_5', 'kw1_6', 'kw1_7', 'kw1_8', 'kw1_9']
kw2_feature = [			'kw2_0', 'kw2_1', 'kw2_2', 'kw2_3', 'kw2_4', 'kw2_5', 'kw2_6', 'kw2_7', 'kw2_8', 'kw2_9']
kw3_feature = [			'kw3_0', 'kw3_1', 'kw3_2', 'kw3_3', 'kw3_4', 'kw3_5', 'kw3_6', 'kw3_7', 'kw3_8', 'kw3_9']
topic1_feature = [		'topic1_0', 'topic1_1', 'topic1_2', 'topic1_3', 'topic1_4', 
						'topic1_5', 'topic1_6', 'topic1_7', 'topic1_8', 'topic1_9']
topic2_feature = [		'topic2_0', 'topic2_1', 'topic2_2', 'topic2_3', 'topic2_4', 
						'topic2_5', 'topic2_6', 'topic2_7', 'topic2_8', 'topic2_9']
topic3_feature = [		'topic3_0', 'topic3_1', 'topic3_2', 'topic3_3', 'topic3_4', 
						'topic3_5', 'topic3_6', 'topic3_7', 'topic3_8', 'topic3_9']

one_hot_feature = [	'productType', 'age', 'gender', 'education', 'consumptionAbility', 'carrier', 'house_01', 'aid',
					'advertiserId', 'campaignId', 'adCategoryId', 'productId', 'marriageStatus_LE', 'ct_LE',
					'os_LE', 'appIdInstall_01', 'appIdAction_01']

continuous_feature = [	'creativeSize','LBS','interest1_0','interest1_1','interest1_2','interest1_3','interest1_4','interest1_5','interest1_6','interest1_7','interest1_8','interest1_9','interest2_0','interest2_1','interest2_2','interest2_3','interest2_4','interest2_5','interest2_6','interest2_7','interest2_8','interest2_9','interest3_0','interest3_1','interest3_2','interest3_3','interest3_4','interest3_5','interest3_6','interest3_7','interest3_8','interest3_9','interest4_0','interest4_1','interest4_2','interest4_3','interest4_4','interest4_5','interest4_6','interest4_7','interest4_8','interest4_9','interest5_0','interest5_1','interest5_2','interest5_3','interest5_4','interest5_5','interest5_6','interest5_7','interest5_8','interest5_9','kw1_0','kw1_1','kw1_2','kw1_3','kw1_4','kw1_5','kw1_6','kw1_7','kw1_8','kw1_9','kw2_0','kw2_1','kw2_2','kw2_3','kw2_4','kw2_5','kw2_6','kw2_7','kw2_8','kw2_9','kw3_0','kw3_1','kw3_2','kw3_3','kw3_4','kw3_5','kw3_6','kw3_7','kw3_8','kw3_9','topic1_0','topic1_1','topic1_2','topic1_3','topic1_4','topic1_5','topic1_6','topic1_7','topic1_8','topic1_9','topic2_0','topic2_1','topic2_2','topic2_3','topic2_4','topic2_5','topic2_6','topic2_7','topic2_8','topic2_9','topic3_0','topic3_1','topic3_2','topic3_3','topic3_4','topic3_5','topic3_6','topic3_7','topic3_8','topic3_9','interest1_length','interest2_length','interest3_length','interest4_length','interest5_length','kw1_length','kw2_length','kw3_length','topic1_length','topic2_length','topic3_length','vector_num','map_advertiserId_campaignId','map_campaignId_aid','aidCount','advertiserIdCount','campaignIdCount','adCategoryIdCount','productIdCount','aid_campaignId','aid_advertiserId','aid_adCategoryId','aid_aidCount','aid_productType','aid_age','aid_gender','aid_education','aid_consumptionAbility','aid_marriageStatus_LE','aid_ct_LE','aid_os_LE','aid_interest1_length','aid_vector_num','aid_interest2_length','aid_interest5_length','campaignId_advertiserId','campaignId_adCategoryId','campaignId_aidCount','campaignId_productType','campaignId_age','campaignId_gender','campaignId_education','campaignId_consumptionAbility','campaignId_marriageStatus_LE','campaignId_ct_LE','campaignId_os_LE','campaignId_interest1_length','campaignId_vector_num','campaignId_interest2_length','campaignId_interest5_length','advertiserId_adCategoryId','advertiserId_aidCount','advertiserId_productType','advertiserId_age','advertiserId_gender','advertiserId_education','advertiserId_consumptionAbility','advertiserId_marriageStatus_LE','advertiserId_ct_LE','advertiserId_os_LE','advertiserId_interest1_length','advertiserId_vector_num','advertiserId_interest2_length','advertiserId_interest5_length','adCategoryId_aidCount','adCategoryId_productType','adCategoryId_age','adCategoryId_gender','adCategoryId_education','adCategoryId_consumptionAbility','adCategoryId_marriageStatus_LE','adCategoryId_ct_LE','adCategoryId_os_LE','adCategoryId_interest1_length','adCategoryId_vector_num','adCategoryId_interest2_length','adCategoryId_interest5_length','aidCount_productType','aidCount_age','aidCount_gender','aidCount_education','aidCount_consumptionAbility','aidCount_marriageStatus_LE','aidCount_ct_LE','aidCount_os_LE','aidCount_interest1_length','aidCount_vector_num','aidCount_interest2_length','aidCount_interest5_length','productType_age','productType_gender','productType_education','productType_consumptionAbility','productType_marriageStatus_LE','productType_ct_LE','productType_os_LE','productType_interest1_length','productType_vector_num','productType_interest2_length','productType_interest5_length','age_gender','age_education','age_consumptionAbility','age_marriageStatus_LE','age_ct_LE','age_os_LE','age_interest1_length','age_vector_num','age_interest2_length','age_interest5_length','gender_education','gender_consumptionAbility','gender_marriageStatus_LE','gender_ct_LE','gender_os_LE','gender_interest1_length','gender_vector_num','gender_interest2_length','gender_interest5_length','education_consumptionAbility','education_marriageStatus_LE','education_ct_LE','education_os_LE','education_interest1_length','education_vector_num','education_interest2_length','education_interest5_length','consumptionAbility_marriageStatus_LE','consumptionAbility_ct_LE','consumptionAbility_os_LE','consumptionAbility_interest1_length','consumptionAbility_vector_num','consumptionAbility_interest2_length','consumptionAbility_interest5_length','marriageStatus_LE_ct_LE','marriageStatus_LE_os_LE','marriageStatus_LE_interest1_length','marriageStatus_LE_vector_num','marriageStatus_LE_interest2_length','marriageStatus_LE_interest5_length','ct_LE_os_LE','ct_LE_interest1_length','ct_LE_vector_num','ct_LE_interest2_length','ct_LE_interest5_length','os_LE_interest1_length','os_LE_vector_num','os_LE_interest2_length','os_LE_interest5_length','interest1_length_vector_num','interest1_length_interest2_length','interest1_length_interest5_length','vector_num_interest2_length','vector_num_interest5_length','interest2_length_interest5_length','aid_campaignId_advertiserId','aid_campaignId_age','aid_campaignId_gender','aid_campaignId_education','aid_campaignId_consumptionAbility','aid_campaignId_marriageStatus_LE','aid_campaignId_ct_LE','aid_campaignId_os_LE','aid_advertiserId_age','aid_advertiserId_gender','aid_advertiserId_education','aid_advertiserId_consumptionAbility','aid_advertiserId_marriageStatus_LE','aid_advertiserId_ct_LE','aid_advertiserId_os_LE','aid_age_gender','aid_age_education','aid_age_consumptionAbility','aid_age_marriageStatus_LE','aid_age_ct_LE','aid_age_os_LE','aid_gender_education','aid_gender_consumptionAbility','aid_gender_marriageStatus_LE','aid_gender_ct_LE','aid_gender_os_LE','aid_education_consumptionAbility','aid_education_marriageStatus_LE','aid_education_ct_LE','aid_education_os_LE','aid_consumptionAbility_marriageStatus_LE','aid_consumptionAbility_ct_LE','aid_consumptionAbility_os_LE','aid_marriageStatus_LE_ct_LE','aid_marriageStatus_LE_os_LE','aid_ct_LE_os_LE','campaignId_advertiserId_age','campaignId_advertiserId_gender','campaignId_advertiserId_education','campaignId_advertiserId_consumptionAbility','campaignId_advertiserId_marriageStatus_LE','campaignId_advertiserId_ct_LE','campaignId_advertiserId_os_LE','campaignId_age_gender','campaignId_age_education','campaignId_age_consumptionAbility','campaignId_age_marriageStatus_LE','campaignId_age_ct_LE','campaignId_age_os_LE','campaignId_gender_education','campaignId_gender_consumptionAbility','campaignId_gender_marriageStatus_LE','campaignId_gender_ct_LE','campaignId_gender_os_LE','campaignId_education_consumptionAbility','campaignId_education_marriageStatus_LE','campaignId_education_ct_LE','campaignId_education_os_LE','campaignId_consumptionAbility_marriageStatus_LE','campaignId_consumptionAbility_ct_LE','campaignId_consumptionAbility_os_LE','campaignId_marriageStatus_LE_ct_LE','campaignId_marriageStatus_LE_os_LE','campaignId_ct_LE_os_LE','advertiserId_age_gender','advertiserId_age_education','advertiserId_age_consumptionAbility','advertiserId_age_marriageStatus_LE','advertiserId_age_ct_LE','advertiserId_age_os_LE','advertiserId_gender_education','advertiserId_gender_consumptionAbility','advertiserId_gender_marriageStatus_LE','advertiserId_gender_ct_LE','advertiserId_gender_os_LE','advertiserId_education_consumptionAbility','advertiserId_education_marriageStatus_LE','advertiserId_education_ct_LE','advertiserId_education_os_LE','advertiserId_consumptionAbility_marriageStatus_LE','advertiserId_consumptionAbility_ct_LE','advertiserId_consumptionAbility_os_LE','advertiserId_marriageStatus_LE_ct_LE','advertiserId_marriageStatus_LE_os_LE','advertiserId_ct_LE_os_LE','age_gender_education','age_gender_consumptionAbility','age_gender_marriageStatus_LE','age_gender_ct_LE','age_gender_os_LE','age_education_consumptionAbility','age_education_marriageStatus_LE','age_education_ct_LE','age_education_os_LE','age_consumptionAbility_marriageStatus_LE','age_consumptionAbility_ct_LE','age_consumptionAbility_os_LE','age_marriageStatus_LE_ct_LE','age_marriageStatus_LE_os_LE','age_ct_LE_os_LE','gender_education_consumptionAbility','gender_education_marriageStatus_LE','gender_education_ct_LE','gender_education_os_LE','gender_consumptionAbility_marriageStatus_LE','gender_consumptionAbility_ct_LE','gender_consumptionAbility_os_LE','gender_marriageStatus_LE_ct_LE','gender_marriageStatus_LE_os_LE','gender_ct_LE_os_LE','education_consumptionAbility_marriageStatus_LE','education_consumptionAbility_ct_LE','education_consumptionAbility_os_LE','education_marriageStatus_LE_ct_LE','education_marriageStatus_LE_os_LE','education_ct_LE_os_LE','consumptionAbility_marriageStatus_LE_ct_LE','consumptionAbility_marriageStatus_LE_os_LE','consumptionAbility_ct_LE_os_LE','marriageStatus_LE_ct_LE_os_LE','aid_CTR','campaignId_CTR','advertiserId_CTR','adCategoryId_CTR','aidCount_CTR','productType_CTR','age_CTR','gender_CTR','education_CTR','consumptionAbility_CTR','marriageStatus_LE_CTR','ct_LE_CTR','os_LE_CTR','interest1_length_CTR','vector_num_CTR','interest2_length_CTR','interest5_length_CTR','kw2_length_CTR','age_marriageStatus_LE_ct_LE_CTR','education_ct_LE_CTR','age_education_marriageStatus_LE_CTR','age_gender_education_CTR','education_marriageStatus_LE_os_LE_CTR','age_education_consumptionAbility_CTR','age_consumptionAbility_marriageStatus_LE_CTR','aidCount_education_CTR','consumptionAbility_marriageStatus_LE_os_LE_CTR','ct_LE_os_LE_CTR','age_gender_os_LE_CTR','age_gender_consumptionAbility_CTR','age_consumptionAbility_os_LE_CTR','aidCount_consumptionAbility_CTR','aid_marriageStatus_LE_ct_LE_CTR','age_education_os_LE_CTR','productType_consumptionAbility_CTR','education_consumptionAbility_os_LE_CTR','aid_age_marriageStatus_LE_CTR','aidCount_os_LE_CTR','aid_age_ct_LE_CTR']

features = one_hot_feature + continuous_feature

# loading data
print('loading data...', flush=True)
data = pd.read_csv(path + 'data_stageB.csv')

print("len_data:", len(data), flush=True)
train = data[:8798814]
len_train = len(train)
len_valid = int(len_train / 5)
len_train -= len_valid
print("len_train:", len_train, flush=True)
print("len_valid:", len_valid, flush=True)
print("len_test:", len(data) - len_train - len_valid, flush=True)
Y = train.pop('label')
data.drop(['label'], inplace=True, axis=1)
del train
gc.collect()

#data = data[one_hot_feature + continuous_feature]

class FFMFormat:
	def __init__(self, one_hot_feat, continuous_feat):
		self.field_index_ = None
		self.feature_index_ = None
		self.one_hot_feat = one_hot_feat
		self.continuous_feat = continuous_feat

	def fit(self, df, y=None):
		#self.field_index_ = {col: i for i, col in enumerate(df.columns)}
		self.field_index_ = {}
		#print(df.columns)
		i = 11
		print("setting field_index_...", flush=True)
		for col in df.columns:
			print(col, flush=True)
			if col in interest1_feature:
				self.field_index_[col] = 0
			elif col in interest2_feature:
				self.field_index_[col] = 1
			elif col in interest3_feature:
				self.field_index_[col] = 2
			elif col in interest4_feature:
				self.field_index_[col] = 3
			elif col in interest5_feature:
				self.field_index_[col] = 4
			elif col in kw1_feature:
				self.field_index_[col] = 5
			elif col in kw2_feature:
				self.field_index_[col] = 6
			elif col in kw3_feature:
				self.field_index_[col] = 7
			elif col in topic1_feature:
				self.field_index_[col] = 8
			elif col in topic2_feature:
				self.field_index_[col] = 9
			elif col in topic3_feature:
				self.field_index_[col] = 10
			else:
				self.field_index_[col] = i
				i += 1
		f = open(path + 'field_index.txt', 'w')
		f.write(str(self.field_index_))
		f.close()
		print("field_index_:", self.field_index_, flush=True)
		print("setting feature_index_...", flush=True)
		self.feature_index_ = dict()
		last_idx = 0
		for col in df.columns:
			if col in self.one_hot_feat:
				# print(col)
				df[col] = df[col].astype('int')
				vals = np.unique(df[col])
				for val in vals:
					if val == -1: continue
					name = '{}_{}'.format(col, int(val))
					if name not in self.feature_index_:
						self.feature_index_[name] = last_idx
						last_idx += 1
			elif col in self.continuous_feat:
				self.feature_index_[col] = last_idx
				last_idx += 1

		f = open(path + 'feature_index.txt', 'w')
		f.write(str(self.feature_index_))
		f.close()
		return self

	def fit_transform(self, df, y=None):
		print("fitting..", flush=True)
		self.fit(df, y)
		print("transforming..", flush=True)
		return self.transform(df)

	def transform_row_(self, row):
		ffm = []

		for col, val in row.loc[row != 0].to_dict().items():
			if col in self.one_hot_feat:
				name = '{}_{}'.format(col, int(val))
				if name in self.feature_index_:
					ffm.append('{}:{}:1'.format(self.field_index_[col], self.feature_index_[name]))
				# ffm.append('{}:{}:{}'.format(self.field_index_[col], self.feature_index_[col], 1))
			elif col in self.continuous_feat:
				if val != -1:
					ffm.append('{}:{}:{}'.format(self.field_index_[col], self.feature_index_[col], val))				
		return ' '.join(ffm)

	def transform(self, df):
		# val=[]
		# for k,v in self.feature_index_.items():
		#     val.append(v)
		# val.sort()
		# print(val)
		# print(self.field_index_)
		# print(self.feature_index_)
		return pd.Series({idx: self.transform_row_(row) for idx, row in df.iterrows()})

tr = FFMFormat(one_hot_feature, continuous_feature)
user_ffm = tr.fit_transform(data)
print("ffm shape:", user_ffm.shape, flush=True)
user_ffm.to_csv(path + 'ffm.csv', index=False)

with open(path + 'ffm.csv') as fin:
	f_train_out = open(path + 'train_ffm.txt', 'w')
	f_valid_out = open(path + 'valid_ffm.txt', 'w')
	f_test_out = open(path + 'test_ffm.txt', 'w')
	
	print('writing to file..', flush=True)
	for (i, line) in enumerate(fin):
		if i < len_train:
			f_train_out.write(str(Y[i]) + ' ' + line)
		elif (i >= len_train and i < (len_train + len_valid)):
			f_valid_out.write(str(Y[i]) + ' ' + line)
		else:
			f_test_out.write(line)
	f_train_out.close()
	f_valid_out.close()
	f_test_out.close()
