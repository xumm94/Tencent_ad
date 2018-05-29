import pandas as pd
import gc

pdF = ['aid', 'uid', 'age', 'gender', 'education', 'consumptionAbility', 'LBS', 'carrier', 'house', 'marriageStatus.1',
       'ct.1', 'os.1', 'interest10', 'interest11', 'interest12', 'interest13', 'interest14', 'interest15', 'interest16',
       'interest17', 'interest18', 'interest19', 'interest20', 'interest21', 'interest22', 'interest23', 'interest24',
       'interest25', 'interest26', 'interest27', 'interest28', 'interest29', 'interest30', 'interest31', 'interest32',
       'interest33', 'interest34', 'interest35', 'interest36', 'interest37', 'interest38', 'interest39', 'interest40',
       'interest41', 'interest42', 'interest43', 'interest44', 'interest45', 'interest46', 'interest47', 'interest48',
       'interest49', 'interest50', 'interest51', 'interest52', 'interest53', 'interest54', 'interest55', 'interest56',
       'interest57', 'interest58', 'interest59', 'kw10', 'kw11', 'kw12', 'kw13', 'kw14', 'kw15', 'kw16', 'kw17', 'kw18',
       'kw19', 'kw20', 'kw21', 'kw22', 'kw23', 'kw24', 'kw25', 'kw26', 'kw27', 'kw28', 'kw29', 'kw30', 'kw31', 'kw32',
       'kw33', 'kw34', 'kw35', 'kw36', 'kw37', 'kw38', 'kw39', 'topic10', 'topic11', 'topic12', 'topic13', 'topic14',
       'topic15', 'topic16', 'topic17', 'topic18', 'topic19', 'topic20', 'topic21', 'topic22', 'topic23', 'topic24',
       'topic25', 'topic26', 'topic27', 'topic28', 'topic29', 'topic30', 'topic31', 'topic32', 'topic33', 'topic34',
       'topic35', 'topic36', 'topic37', 'topic38', 'topic39']
e1F = ['appIdInstall.1', 'appIdAction.1', 'uidCount', 'uidCount_age', 'uidCount_gender', 'uidCount_education',
       'uidCount_consumptionAbility', 'uidCount_appIdInstall.1', 'uidCount_carrier', 'uidCount_house',
       'uidCount_marriageStatus.1', 'uidCount_ct.1', 'uidCount_os.1', 'age_gender', 'age_education', 'age_consumptionAbility',
       'age_appIdInstall.1', 'age_carrier', 'age_house', 'age_marriageStatus.1', 'age_ct.1', 'age_os.1', 'gender_education',
       'gender_consumptionAbility', 'gender_appIdInstall.1', 'gender_carrier', 'gender_house', 'gender_marriageStatus.1',
       'gender_ct.1', 'gender_os.1', 'education_consumptionAbility', 'education_appIdInstall.1', 'education_carrier',
       'education_house', 'education_marriageStatus.1', 'education_ct.1', 'education_os.1', 'consumptionAbility_appIdInstall.1',
       'consumptionAbility_carrier', 'consumptionAbility_house', 'consumptionAbility_marriageStatus.1', 'consumptionAbility_ct.1',
       'consumptionAbility_os.1', 'appIdInstall.1_carrier', 'appIdInstall.1_house', 'appIdInstall.1_marriageStatus.1',
       'appIdInstall.1_ct.1', 'appIdInstall.1_os.1', 'carrier_house', 'carrier_marriageStatus.1', 'carrier_ct.1', 'carrier_os.1',
       'house_marriageStatus.1', 'house_ct.1', 'house_os.1', 'marriageStatus.1_ct.1', 'marriageStatus.1_os.1', 'ct.1_os.1',
       'age_gender_education', 'age_gender_consumptionAbility', 'age_gender_house', 'age_gender_marriageStatus.1',
       'age_gender_os.1', 'age_education_consumptionAbility', 'age_education_house', 'age_education_marriageStatus.1',
       'age_education_os.1', 'age_consumptionAbility_house', 'age_consumptionAbility_marriageStatus.1', 'age_consumptionAbility_os.1',
       'age_house_marriageStatus.1', 'age_house_os.1', 'age_marriageStatus.1_os.1', 'gender_education_consumptionAbility',
       'gender_education_house', 'gender_education_marriageStatus.1', 'gender_education_os.1', 'gender_consumptionAbility_house',
       'gender_consumptionAbility_marriageStatus.1', 'gender_consumptionAbility_os.1', 'gender_house_marriageStatus.1',
       'gender_house_os.1', 'gender_marriageStatus.1_os.1', 'education_consumptionAbility_house',
       'education_consumptionAbility_marriageStatus.1', 'education_consumptionAbility_os.1', 'education_house_marriageStatus.1',
       'education_house_os.1', 'education_marriageStatus.1_os.1', 'consumptionAbility_house_marriageStatus.1',
       'consumptionAbility_house_os.1', 'consumptionAbility_marriageStatus.1_os.1', 'house_marriageStatus.1_os.1']
e2F = ['ct.1_marriageStatus.1', 'ct.1_age', 'marriageStatus.1_age', 'ct.1_marriageStatus.1_age']

# set file path
path = '../preliminary_contest_data/'

print('loading aid...')
aidList = pd.read_csv(path + 'adFeature.csv', usecols=['aid'])
aidList = aidList['aid'].tolist()
print('loading label...')
label = pd.read_csv(path + 'train.csv', usecols=['label'])
print('loading processed data...')
data = pd.read_csv(path + 'processed_data.csv', usecols=pdF)
print('loading engineered features...')
fe = pd.read_csv(path + 'engineered_features.csv', usecols=e1F)
fe2 = pd.read_csv(path + 'engineered_features_2.csv', usecols=e2F)
print('concatenating...')
data = pd.concat([label, data, fe, fe2], axis=1)
del label
del fe
del fe2
gc.collect()

data.loc[data['label'] == -1, 'label'] = 0
data.fillna(-1, inplace=True)
for aid in aidList:
    print('splitting aid=' + str(aid) + '...')
    temp = data[data['aid'] == aid]
    temp.to_csv(path + 'split/aid' + str(aid) + '.csv', index=False)
print('aid successfully split!')
