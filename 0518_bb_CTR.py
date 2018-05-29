import pandas as pd
import gc
from file_info import processedCols, fe1Cols, fe2Cols, feffCols

path = '../preliminary_contest_data/'  # set file path
len_train = 8798814  # training set size
len_test = 2265989  # test set size

singleF = [
            'aid',
            'ct.1',
            'campaignId',
            'advertiserId',
            'marriageStatus.1',
            'interest1_length',
            'interest2_length',
            'age',
            'vector_num',
            'kw2_length',
            'interest5_length',
            'education',
            'adCategoryId'
        ]

groupF = [
            'productType_age',
            'uidCount_age',
            'age_gender_consumptionAbility',
            'uidCount_consumptionAbility',
            'aidCount_gender',
            'age_education_marriageStatus.1',
            'aidCount_age',
            'ct.1_marriageStatus.1_age',
            'age_gender_os.1',
            'age_consumptionAbility_house',
            'ct.1_os.1',
            'age_consumptionAbility_os.1',
            'age_gender',
            'age_gender_education',
            'age_education_consumptionAbility',
            'advertiserId_age'
        ]

features = singleF + groupF

print('loading from processed data...')
data = pd.read_csv(path + 'processed_data.csv', usecols=list(set(features).intersection(set(processedCols))) + ['label'])

print('loading from engineered features...')
temp = pd.read_csv(path + 'engineered_features.csv', usecols=list(set(features).intersection(set(fe1Cols))))
print('concatenating...')
data = pd.concat([data, temp], axis=1)
del temp
gc.collect()

print('loading from engineered features 2...')
temp = pd.read_csv(path + 'engineered_features_2.csv', usecols=list(set(features).intersection(set(fe2Cols))))
print('concatenating...')
data = pd.concat([data, temp], axis=1)
del temp
gc.collect()

print('loading from fe ff...')
temp = pd.read_csv(path + 'fe_ff.csv', usecols=list(set(features).intersection(set(feffCols))))
print('concatenating...')
data = pd.concat([data, temp], axis=1)
del temp
gc.collect()

print('splitting...')
train = data[:len_train]
test = data[len_train:]
del data
gc.collect()

for i, f in enumerate(features):
    print('processing {}...  {} / {}'.format(f, i + 1, len(features)))
    fVC = train[f].value_counts()
    temp = pd.DataFrame({f: fVC.index,
                         f + '_CTR': [train.loc[(train[f] == fVC.index[i]) & (train['label'] == 1)]['label'].count() * 1.0 / fVC.values[i]
                                      for i in range(len(fVC.index))]})
    print('merging...')
    train = train.merge(temp, on=f, how='left')
    test = test.merge(temp, on=f, how='left')
    del temp
    gc.collect()

print('post-processing...')
train = train.append(test)
train.drop(features + ['label'], axis=1, inplace=True)
del test
gc.collect()
train = train.astype('float16')
print(train.info())

print('writing...')
train.to_csv(path + 'CTR_features.csv', index=False)
del train
gc.collect()
print('done!')
