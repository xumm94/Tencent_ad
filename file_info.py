##################################################
#                    原始数据                     #
##################################################

# train.csv
# 行数：8798815 （包括标题行）
# 列数：3
trainCols = ['aid', 'uid', 'label']

# test1.csv
# 行数：2265990 （包括标题行）
# 列数：2
test1Cols = ['aid', 'uid']

# adFeature.csv
# 行数：174 （包括标题行）
# 列数：8
adFCols = ['aid', 'advertiserId', 'campaignId', 'creativeId', 'creativeSize', 'adCategoryId', 'productId', 'productType']

# userFeature.csv
# 行数：11420040 （包括标题行）
# 列数：24
userFCols = ['uid', 'age', 'gender', 'marriageStatus', 'education', 'consumptionAbility', 'LBS',
             'interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3',
             'topic1', 'topic2', 'topic3', 'appIdInstall', 'appIdAction', 'ct', 'os', 'carrier', 'house']


##################################################
#                   预处理数据                    #
##################################################

# processed_data.csv
# 来源：上下拼接train.csv与test1.csv，根据aid、uid merge全表，对marriageStatus、ct、os进行LabelEncoder处理，得到相应feature.1特征；
# 对interest、kw、topic、appIdInstall、appIdAction进行w2v处理，得到相应feature0-feature9特征
# 行数：11064804 （包括标题行）
# 列数：165
processedCols = ['aid', 'label', 'uid', 'advertiserId', 'campaignId',
                 'creativeSize', 'adCategoryId', 'productId', 'productType', 'age',
                 'gender', 'marriageStatus', 'education', 'consumptionAbility',
                 'LBS', 'interest1', 'interest2', 'interest3', 'interest4',
                 'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3',
                 'appIdInstall', 'appIdAction', 'ct', 'os', 'carrier', 'house',
                 'marriageStatus.1', 'ct.1', 'os.1', 'interest10', 'interest11',
                 'interest12', 'interest13', 'interest14', 'interest15',
                 'interest16', 'interest17', 'interest18', 'interest19',
                 'interest20', 'interest21', 'interest22', 'interest23',
                 'interest24', 'interest25', 'interest26', 'interest27',
                 'interest28', 'interest29', 'interest30', 'interest31',
                 'interest32', 'interest33', 'interest34', 'interest35',
                 'interest36', 'interest37', 'interest38', 'interest39',
                 'interest40', 'interest41', 'interest42', 'interest43',
                 'interest44', 'interest45', 'interest46', 'interest47',
                 'interest48', 'interest49', 'interest50', 'interest51',
                 'interest52', 'interest53', 'interest54', 'interest55',
                 'interest56', 'interest57', 'interest58', 'interest59', 'kw10',
                 'kw11', 'kw12', 'kw13', 'kw14', 'kw15', 'kw16', 'kw17', 'kw18',
                 'kw19', 'kw20', 'kw21', 'kw22', 'kw23', 'kw24', 'kw25', 'kw26',
                 'kw27', 'kw28', 'kw29', 'kw30', 'kw31', 'kw32', 'kw33', 'kw34',
                 'kw35', 'kw36', 'kw37', 'kw38', 'kw39', 'topic10', 'topic11',
                 'topic12', 'topic13', 'topic14', 'topic15', 'topic16', 'topic17',
                 'topic18', 'topic19', 'topic20', 'topic21', 'topic22', 'topic23',
                 'topic24', 'topic25', 'topic26', 'topic27', 'topic28', 'topic29',
                 'topic30', 'topic31', 'topic32', 'topic33', 'topic34', 'topic35',
                 'topic36', 'topic37', 'topic38', 'topic39', 'appIdInstall0',
                 'appIdInstall1', 'appIdInstall2', 'appIdInstall3', 'appIdInstall4',
                 'appIdInstall5', 'appIdInstall6', 'appIdInstall7', 'appIdInstall8',
                 'appIdInstall9', 'appIdAction0', 'appIdAction1', 'appIdAction2',
                 'appIdAction3', 'appIdAction4', 'appIdAction5', 'appIdAction6',
                 'appIdAction7', 'appIdAction8', 'appIdAction9']


##################################################
#                  特征工程数据                    #
##################################################

# engineered_features.csv
# 来源：对appIdInstall、appIdAction进行0-1处理，有值为1缺省为0；对id特征进行计数处理，得到相应featureCount特征；对部分特征进行组合计数处理，
# 得到相应feature1_feature2或feature1_feature2_feature3特征
# 使用：保持train.csv与test1.csv拼接顺序，可直接concat
# 行数：11064804 （包括标题行）
# 列数：121
fe1Cols = ['appIdInstall.1', 'appIdAction.1', 'aidCount', 'advertiserIdCount',
           'campaignIdCount', 'adCategoryIdCount', 'productIdCount',
           'uidCount', 'aidCount_uidCount', 'aidCount_productType',
           'aidCount_age', 'aidCount_gender', 'aidCount_education',
           'aidCount_consumptionAbility', 'aidCount_appIdInstall.1',
           'aidCount_carrier', 'aidCount_house', 'aidCount_marriageStatus.1',
           'aidCount_ct.1', 'aidCount_os.1', 'uidCount_productType',
           'uidCount_age', 'uidCount_gender', 'uidCount_education',
           'uidCount_consumptionAbility', 'uidCount_appIdInstall.1',
           'uidCount_carrier', 'uidCount_house', 'uidCount_marriageStatus.1',
           'uidCount_ct.1', 'uidCount_os.1', 'productType_age',
           'productType_gender', 'productType_education',
           'productType_consumptionAbility', 'productType_appIdInstall.1',
           'productType_carrier', 'productType_house',
           'productType_marriageStatus.1', 'productType_ct.1',
           'productType_os.1', 'age_gender', 'age_education',
           'age_consumptionAbility', 'age_appIdInstall.1', 'age_carrier',
           'age_house', 'age_marriageStatus.1', 'age_ct.1', 'age_os.1',
           'gender_education', 'gender_consumptionAbility',
           'gender_appIdInstall.1', 'gender_carrier', 'gender_house',
           'gender_marriageStatus.1', 'gender_ct.1', 'gender_os.1',
           'education_consumptionAbility', 'education_appIdInstall.1',
           'education_carrier', 'education_house',
           'education_marriageStatus.1', 'education_ct.1', 'education_os.1',
           'consumptionAbility_appIdInstall.1', 'consumptionAbility_carrier',
           'consumptionAbility_house', 'consumptionAbility_marriageStatus.1',
           'consumptionAbility_ct.1', 'consumptionAbility_os.1',
           'appIdInstall.1_carrier', 'appIdInstall.1_house',
           'appIdInstall.1_marriageStatus.1', 'appIdInstall.1_ct.1',
           'appIdInstall.1_os.1', 'carrier_house', 'carrier_marriageStatus.1',
           'carrier_ct.1', 'carrier_os.1', 'house_marriageStatus.1',
           'house_ct.1', 'house_os.1', 'marriageStatus.1_ct.1',
           'marriageStatus.1_os.1', 'ct.1_os.1', 'age_gender_education',
           'age_gender_consumptionAbility', 'age_gender_house',
           'age_gender_marriageStatus.1', 'age_gender_os.1',
           'age_education_consumptionAbility', 'age_education_house',
           'age_education_marriageStatus.1', 'age_education_os.1',
           'age_consumptionAbility_house',
           'age_consumptionAbility_marriageStatus.1',
           'age_consumptionAbility_os.1', 'age_house_marriageStatus.1',
           'age_house_os.1', 'age_marriageStatus.1_os.1',
           'gender_education_consumptionAbility', 'gender_education_house',
           'gender_education_marriageStatus.1', 'gender_education_os.1',
           'gender_consumptionAbility_house',
           'gender_consumptionAbility_marriageStatus.1',
           'gender_consumptionAbility_os.1', 'gender_house_marriageStatus.1',
           'gender_house_os.1', 'gender_marriageStatus.1_os.1',
           'education_consumptionAbility_house',
           'education_consumptionAbility_marriageStatus.1',
           'education_consumptionAbility_os.1',
           'education_house_marriageStatus.1', 'education_house_os.1',
           'education_marriageStatus.1_os.1',
           'consumptionAbility_house_marriageStatus.1',
           'consumptionAbility_house_os.1',
           'consumptionAbility_marriageStatus.1_os.1',
           'house_marriageStatus.1_os.1']

# engineered_features_2.csv
# 来源：对部分强特征进行组合计数处理，得到相应feature1_feature2或feature1_feature2_feature3特征
# 使用：保持train.csv与test1.csv拼接顺序，可直接concat
# 行数：11064804 （包括标题行）
# 列数：35
fe2Cols = ['aid_ct.1', 'aid_campaignId', 'aid_advertiserId',
           'aid_marriageStatus.1', 'aid_age', 'ct.1_campaignId',
           'ct.1_advertiserId', 'ct.1_marriageStatus.1', 'ct.1_age',
           'campaignId_advertiserId', 'campaignId_marriageStatus.1',
           'campaignId_age', 'advertiserId_marriageStatus.1',
           'advertiserId_age', 'marriageStatus.1_age', 'aid_ct.1_campaignId',
           'aid_ct.1_advertiserId', 'aid_ct.1_marriageStatus.1',
           'aid_ct.1_age', 'aid_campaignId_advertiserId',
           'aid_campaignId_marriageStatus.1', 'aid_campaignId_age',
           'aid_advertiserId_marriageStatus.1', 'aid_advertiserId_age',
           'aid_marriageStatus.1_age', 'ct.1_campaignId_advertiserId',
           'ct.1_campaignId_marriageStatus.1', 'ct.1_campaignId_age',
           'ct.1_advertiserId_marriageStatus.1', 'ct.1_advertiserId_age',
           'ct.1_marriageStatus.1_age',
           'campaignId_advertiserId_marriageStatus.1',
           'campaignId_advertiserId_age', 'campaignId_marriageStatus.1_age',
           'advertiserId_marriageStatus.1_age']

# fe_ff.csv
# 来源：统计了各向量特征长度以及总长度，反映用户填写完整程度；统计了广告主-推广计划-广告之间的层级对应数量关系
# 使用：保持train.csv与test1.csv拼接顺序，可直接concat
# 行数：11064804 （包括标题行）
# 列数：14
feffCols = ['interest1_length', 'interest2_length', 'interest3_length', 'interest4_length', 'interest5_length',
            'kw1_length', 'kw2_length', 'kw3_length', 'topic1_length', 'topic2_length', 'topic3_length', 'vector_num',
            'map_advertiserId_campaignId', 'map_campaignId_aid']

# nlp_features.csv
# 来源：https://github.com/LightR0/Tencent_Ads_2018
# 使用：保持train.csv与test1.csv拼接顺序，可直接concat
# 行数：11064804 （包括标题行）
# 列数：10
nlpCols = ['appIdInstall_score', 'appIdAction_score', 'interest1_score', 'interest2_score', 'interest5_score',
           'kw1_score', 'kw2_score', 'topic1_score', 'topic2_score', 'campaignId_active_aid']

# w2v_features.csv
# 来源：从processed_data.csv中提取的w2v特征
# 使用：保持train.csv与test1.csv拼接顺序，可直接concat
# 行数：11064804 （包括标题行）
# 列数：56

w2vCols = ['kw21', 'kw26', 'kw25', 'kw23', 'kw20', 'topic24', 'kw28', 'kw22', 'kw27', 'kw24',
           'topic20', 'topic27', 'kw29', 'topic23', 'topic26', 'interest12', 'interest26', 'interest21',
           'kw18', 'topic28', 'topic16', 'topic22', 'interest28', 'topic14', 'kw17', 'topic25', 'interest27',
           'interest23', 'interest25', 'kw16', 'topic21', 'interest59', 'topic10', 'kw10', 'interest11',
           'interest29', 'kw19', 'topic29', 'kw14', 'kw12', 'interest51', 'topic11', 'interest17', 'kw13',
           'interest53', 'topic19', 'kw11', 'topic17', 'interest20', 'topic15', 'interest55', 'kw15',
           'topic13', 'interest13', 'interest10', 'interest24']

# CTR_features.csv
# 来源：提取部分重要性高的特征的转化率，得到相应feature_CTR特征
# 使用：保持train.csv与test1.csv拼接顺序，可直接concat
# 行数：11064804 （包括标题行）
# 列数：29

ctrCols = ['aid_CTR', 'ct.1_CTR', 'campaignId_CTR', 'advertiserId_CTR',
           'marriageStatus.1_CTR', 'interest1_length_CTR', 'interest2_length_CTR',
           'age_CTR', 'vector_num_CTR', 'kw2_length_CTR', 'interest5_length_CTR',
           'education_CTR', 'adCategoryId_CTR', 'productType_age_CTR',
           'uidCount_age_CTR', 'age_gender_consumptionAbility_CTR',
           'uidCount_consumptionAbility_CTR', 'aidCount_gender_CTR',
           'age_education_marriageStatus.1_CTR', 'aidCount_age_CTR',
           'ct.1_marriageStatus.1_age_CTR', 'age_gender_os.1_CTR',
           'age_consumptionAbility_house_CTR', 'ct.1_os.1_CTR',
           'age_consumptionAbility_os.1_CTR', 'age_gender_CTR',
           'age_gender_education_CTR', 'age_education_consumptionAbility_CTR',
           'advertiserId_age_CTR']
