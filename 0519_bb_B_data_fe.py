from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import gc
from multiprocessing import Pool
from scipy.special import comb


##################################################
#          I/O with dtypes set                   #
##################################################
def dtypes_to_csv(df, path):
    df.loc[-1] = df.dtypes
    df[-1:].to_csv(path, index=False)
    df.drop(-1, inplace=True)


def dtypes_read_csv(path):
    return pd.read_csv(path).iloc[0].to_dict()


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
    print('processing', feature, flush=True)
    data[feature] = data[feature].apply(lambda x: str(x).split(' '))
    model = Word2Vec(data[feature], size=10, min_count=1, iter=5, window=2)
    data_vec = []
    for row in data[feature]:
        data_vec.append(base_word2vec(row, model, size=10))
    column_names = []
    for i in range(10):
        column_names.append(feature + '_' + str(i))
    data_vec = pd.DataFrame(data_vec, columns=column_names)
    data_vec = data_vec.astype('float16')
    return data_vec


##################################################
#          one-hot feature pre-process           #
##################################################
# Method 1: using LabelEncoder
#
#            ？？？是不是去重更好？？？
#
def one_hot_feature_process(feature, encoder):
    global data
    print('processing', feature, flush=True)
    try:
        data[feature] = data[feature].apply(lambda x: str(x).split(' '))
        curr_col = encoder.fit_transform(data[feature].apply(int))
    except:
        curr_col = encoder.fit_transform(data[feature])
    curr_col = pd.DataFrame(curr_col)
    curr_col.columns = [feature + '_LE']
    curr_col = curr_col.astype('uint8')
    return curr_col


##################################################
#          feature pre-process                   #
##################################################
# processing vector & one-hot features
def feature_pre_process(one_hot_features, vector_features):
    global data
    n = 8
    new_features = []

    print('processing one-hot features...', flush=True)
    p = Pool(n)
    # encoder = OneHotEncoder()
    encoder = LabelEncoder()
    for feature in one_hot_features:
        new_features.append(p.apply_async(one_hot_feature_process, args=(feature, encoder,)))
    p.close()
    p.join()

    print('concatenating...', flush=True)
    for new_feature in new_features:
        data = pd.concat([data, new_feature.get()], axis=1)
    new_features.clear()
    gc.collect()
    print('one-hot done!', flush=True)

    print('processing w2v features...', flush=True)
    q = Pool(n)
    for feature in vector_features:
        new_features.append(q.apply_async(w2v_feature_process, args=(feature,)))
    q.close()
    q.join()

    print('concatenating...', flush=True)
    for new_feature in new_features:
        data = pd.concat([data, new_feature.get()], axis=1)
    del new_features
    gc.collect()
    print('w2v done!', flush=True)


##################################################
#          feature engineering by feephy         #
##################################################
def get_vector_length(vector):
    if vector == '-1':
        return 0
    else:
        return len(vector.split(' '))


def map_value_count(df, parent, child):
    dfmap = df[child].groupby(df[parent]).unique().apply(lambda x: len(x))
    df['map_' + parent + '_' + child] = df[parent].apply(lambda x: dfmap[x])
    df['map_' + parent + '_' + child] = df['map_' + parent + '_' + child].astype('uint8')


def feephy_feature_engineering(df, vector_features):
    ad_features = ['advertiserId', 'campaignId', 'aid']
    vector_features_length = [feature + '_length' for feature in vector_features]
    # 每个向量特征填了几个
    for feature in vector_features:
        df[feature + '_length'] = df[feature].apply(lambda x: get_vector_length(str(x)))
        df[feature + '_length'] = df[feature + '_length'].astype('uint8')
    # 向量特征总共填了几个
    df['vector_num'] = df[vector_features_length].apply(lambda x: np.sum(x), axis=1)
    df['vector_num'] = df['vector_num'].astype('uint8')
    # 广告层级对应数量关系
    for i in range(2):
        map_value_count(df, ad_features[i], ad_features[i + 1])


##################################################
#          main function                         #
##################################################
if __name__ == '__main__':
    print('############################################################\n', flush=True)
    print('           Tencent Algo Competition    by HBZ               \n', flush=True)
    print('############################################################\n', flush=True)

    one_hot_features = ['marriageStatus', 'ct', 'os']
    vector_features = ['interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
    categorical_features = ['productType', 'age', 'gender', 'education', 'consumptionAbility', 'carrier', 'house']
    app_features = ['appIdInstall', 'appIdAction']
    id_features = ['aid', 'advertiserId', 'campaignId', 'adCategoryId', 'productId', 'uid']
    numeric_features = ['creativeSize', 'LBS']

    path = '../preliminary_contest_data/'  # set file path
    len_train = 8798814  # training set size
    len_test = 2265879  # test set size


##################################################
#          processing original data              #
##################################################
    # extracting features from original datasets
    print('loading train...', flush=True)
    train_dtypes = {'aid': 'uint16', 'uid': 'uint32', 'label': 'int8'}
    data = pd.read_csv(path + "train.csv", dtype=train_dtypes)
    print('loading test...', flush=True)
    test_dtypes = {'aid': 'uint16', 'uid': 'uint32'}
    test = pd.read_csv(path + "test2.csv", dtype=test_dtypes)
    print('loading adFeature...', flush=True)
    adF_dtypes = {'aid': 'uint16', 'advertiserId': 'uint32', 'campaignId': 'uint32', 'creativeSize': 'uint16', 'adCategoryId': 'uint16', 'productId': 'uint16', 'productType': 'uint8'}
    adF = pd.read_csv(path + "adFeature.csv", usecols=['aid', 'advertiserId', 'campaignId', 'creativeSize', 'adCategoryId', 'productId', 'productType'], dtype=adF_dtypes)
    print('loading userFeature...', flush=True)
    userF_dtypes = {'uid': 'uint32', 'age': 'uint8', 'gender': 'uint8', 'education': 'uint8', 'consumptionAbility': 'uint8', 'LBS': 'float16', 'carrier':'uint8', 'house': 'float16'}
    userF = pd.read_csv(path + "userFeature.csv", dtype=userF_dtypes)

    print('\n---------------data pre-processing---------------', flush=True)
    data = data.append(test)
    del test
    gc.collect()

    print('merging...', flush=True)
    data = data.merge(adF, on='aid', how='left')
    data = data.merge(userF, on='uid', how='left')
    del adF
    del userF
    gc.collect()

    data.fillna('-1', inplace=True)
    data['LBS'] = data['LBS'].astype('float16', errors='ignore')

    print('processing app features and house...', flush=True)
    for feature in app_features + ['house']:
        mask = (data[feature] != '-1').rename(feature + '_01').apply(int)
        mask = mask.astype('uint8')
        data = pd.concat([data, mask], axis=1)
    data.drop(app_features + ['house'], axis=1, inplace=True)
    del mask
    gc.collect()
    print('app features and house done!', flush=True)

    print('extracting str features...', flush=True)
    feature_pre_process(one_hot_features, vector_features)


##################################################
#          feature engineering                   #
##################################################
    print('\n---------------feature engineering---------------', flush=True)
    # adding feephy features
    print('adding feephy features...', flush=True)
    feephy_feature_engineering(data, vector_features)
    data.drop(one_hot_features + vector_features, axis=1, inplace=True)
    gc.collect()
    print('feephy features done!', flush=True)

    train = data[:len_train]
    test = data[len_train:]
    train['label'] = train['label'].astype('int8')
    test['label'] = 0
    test['label'] = test['label'].astype('int8')
    del data
    gc.collect()

    # adding count features
    print('adding count feature of ids...', flush=True)
    for id_feature in id_features:
        if id_feature == 'uid':
            continue
        idCount = train[id_feature].value_counts()
        idCount = pd.DataFrame({id_feature: idCount.index, id_feature + 'Count': idCount.values})
        idCount[id_feature] = idCount[id_feature].astype(train[id_feature].dtype)
        idCount[id_feature + 'Count'] = idCount[id_feature + 'Count'].astype('uint32')
        train = train.merge(idCount, on=id_feature, how='left')
        test = test.merge(idCount, on=id_feature, how='left')
        test[id_feature + 'Count'].fillna(0, inplace=True)
        test[id_feature + 'Count'] = test[id_feature + 'Count'].astype('uint32')
    del idCount
    gc.collect()
    print('count of ids done!', flush=True)


    # adding group count features
    print('adding group count features...', flush=True)
    gcf = ['aid', 'campaignId', 'advertiserId', 'adCategoryId', 'aidCount', 'productType', 'age', 'gender', 'education',
           'consumptionAbility', 'marriageStatus_LE', 'ct_LE', 'os_LE', 'interest1_length', 'vector_num',
           'interest2_length', 'interest5_length']
    n, total = 0, int(comb(len(gcf), 2))
    for i in range(len(gcf) - 1):
        for j in range(i + 1, len(gcf)):
            n += 1
            print('{} / {}  grouping {} and {}'.format(n, total, gcf[i], gcf[j]), flush=True)
            gp = train[[gcf[i], gcf[j], 'uid']].groupby(by=[gcf[i], gcf[j]])[['uid']].count().reset_index().rename(index=str, columns={'uid': gcf[i] + '_' + gcf[j]})
            gp[gcf[i]] = gp[gcf[i]].astype(train[gcf[i]].dtype)
            gp[gcf[j]] = gp[gcf[j]].astype(train[gcf[j]].dtype)
            gp[gcf[i] + '_' + gcf[j]] = gp[gcf[i] + '_' + gcf[j]].astype('uint32')
            train = train.merge(gp, on=[gcf[i], gcf[j]], how='left')
            test = test.merge(gp, on=[gcf[i], gcf[j]], how='left')
            test[gcf[i] + '_' + gcf[j]].fillna(0, inplace=True)
            test[gcf[i] + '_' + gcf[j]] = test[gcf[i] + '_' + gcf[j]].astype('uint32')
            del gp
            gc.collect()

    gcf = ['aid', 'campaignId', 'advertiserId', 'age', 'gender', 'education', 'consumptionAbility', 'marriageStatus_LE', 'ct_LE', 'os_LE']
    n, total = 0, int(comb(len(gcf), 3))
    for i in range(len(gcf) - 2):
        for j in range(i + 1, len(gcf) - 1):
            for k in range(j + 1, len(gcf)):
                n += 1
                print('{} / {}  grouping {}, {} and {}'.format(n, total, gcf[i], gcf[j], gcf[k]), flush=True)
                gp = train[[gcf[i], gcf[j], gcf[k], 'uid']].groupby(by=[gcf[i], gcf[j], gcf[k]])[['uid']].count().reset_index().rename(index=str, columns={'uid': gcf[i] + '_' + gcf[j] + '_' + gcf[k]})
                gp[gcf[i]] = gp[gcf[i]].astype(train[gcf[i]].dtype)
                gp[gcf[j]] = gp[gcf[j]].astype(train[gcf[j]].dtype)
                gp[gcf[k]] = gp[gcf[k]].astype(train[gcf[k]].dtype)
                gp[gcf[i] + '_' + gcf[j] + '_' + gcf[k]] = gp[gcf[i] + '_' + gcf[j] + '_' + gcf[k]].astype('uint32')
                train = train.merge(gp, on=[gcf[i], gcf[j], gcf[k]], how='left')
                test = test.merge(gp, on=[gcf[i], gcf[j], gcf[k]], how='left')
                test[gcf[i] + '_' + gcf[j] + '_' + gcf[k]].fillna(0, inplace=True)
                test[gcf[i] + '_' + gcf[j] + '_' + gcf[k]] = test[gcf[i] + '_' + gcf[j] + '_' + gcf[k]].astype('uint32')
                del gp
                gc.collect()
    print('group count features done!', flush=True)

    # adding CTR features
    print('adding CTR features...', flush=True)
    singleF = ['aid', 'campaignId', 'advertiserId', 'adCategoryId', 'aidCount', 'productType', 'age', 'gender', 'education',
               'consumptionAbility', 'marriageStatus_LE', 'ct_LE', 'os_LE', 'interest1_length', 'vector_num', 'interest2_length',
               'interest5_length', 'kw2_length']
    groupF = ['age_marriageStatus_LE_ct_LE', 'education_ct_LE', 'age_education_marriageStatus_LE', 'age_gender_education',
              'education_marriageStatus_LE_os_LE', 'age_education_consumptionAbility', 'age_consumptionAbility_marriageStatus_LE',
              'aidCount_education', 'consumptionAbility_marriageStatus_LE_os_LE', 'ct_LE_os_LE', 'age_gender_os_LE',
              'age_gender_consumptionAbility', 'age_consumptionAbility_os_LE', 'aidCount_consumptionAbility',
              'aid_marriageStatus_LE_ct_LE', 'age_education_os_LE', 'productType_consumptionAbility',
              'education_consumptionAbility_os_LE', 'aid_age_marriageStatus_LE', 'aidCount_os_LE', 'aid_age_ct_LE']
    ctrF = singleF + groupF

    for i, f in enumerate(ctrF):
        print('{} / {}  processing {}'.format(i + 1, len(ctrF), f), flush=True)
        fVC = train[f].value_counts()
        ctr = pd.DataFrame({f: fVC.index,
                             f + '_CTR': [train.loc[(train[f] == fVC.index[i]) & (train['label'] == 1)]['label'].count() * 1.0 / fVC.values[i]
                                          for i in range(len(fVC.index))]})
        ctr[f] = ctr[f].astype(train[f].dtype)
        ctr[f + '_CTR'] = ctr[f + '_CTR'].astype('float16')
        train = train.merge(ctr, on=f, how='left')
        test = test.merge(ctr, on=f, how='left')
        test[f + '_CTR'].fillna(0, inplace=True)
        del ctr
        gc.collect()
    print('CTR features done!', flush=True)

    # adding NLP features
    pass


##################################################
#          writing to local file                 #
##################################################
    print('\n---------------data post-processing---------------', flush=True)
    train = train.append(test)
    del test
    gc.collect()
    print(train.info(), flush=True)

    print('writing to data_stageB.csv...', flush=True)
    # dtypes_to_csv(train, path + 'data_stageB_dtypes.csv')
    train.to_csv(path + 'data_stageB.csv', index=False)
    del train
    gc.collect()
    print('all done!\n', flush=True)
