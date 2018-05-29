from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import HashingVectorizer
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import lightgbm as lgb
import os
import gc
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
# Method 1: using LabelEncoder
def one_hot_feature_process(feature, encoder):
    global data
    print('processing', feature)
    try:
        data[feature] = data[feature].apply(lambda x: str(x).split(' '))
        curr_col = encoder.fit_transform(data[feature].apply(int))
    except:
        curr_col = encoder.fit_transform(data[feature])
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
    # encoder = OneHotEncoder()
    encoder = LabelEncoder()
    for feature in one_hot_features:
        new_features.append(p.apply_async(one_hot_feature_process, args=(feature, encoder,)))
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


##################################################
#          main function                         #
##################################################
if __name__ == '__main__':
    print('############################################################\n')
    print('           Tencent Algo Competition    by HBZ               \n')
    print('############################################################\n')

    one_hot_features = ['marriageStatus', 'ct', 'os']
    vector_features = ['interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1',
                       'topic2', 'topic3']
    categorical_features = ['productType', 'age', 'gender', 'education', 'consumptionAbility', 'carrier', 'house']

    app_features = ['appIdInstall', 'appIdAction']
    id_features = ['aid', 'advertiserId', 'campaignId', 'adCategoryId', 'productId', 'uid']
    numeric_features = ['creativeSize', 'LBS']

    features = ['aid', 'label', 'uid', 'advertiserId', 'campaignId', 'creativeSize', 'adCategoryId', 'productId',
                'productType', 'age', 'gender', 'education', 'consumptionAbility', 'LBS', 'appIdInstall', 'appIdAction',
                'carrier', 'house', 'marriageStatus.1', 'ct.1', 'os.1', 'interest10', 'interest11', 'interest12',
                'interest13', 'interest14', 'interest15', 'interest16', 'interest17', 'interest18', 'interest19',
                'interest20', 'interest21', 'interest22', 'interest23', 'interest24', 'interest25', 'interest26',
                'interest27', 'interest28', 'interest29', 'interest30', 'interest31', 'interest32', 'interest33',
                'interest34', 'interest35', 'interest36', 'interest37', 'interest38', 'interest39', 'interest40',
                'interest41', 'interest42', 'interest43', 'interest44', 'interest45', 'interest46', 'interest47',
                'interest48', 'interest49', 'interest50', 'interest51', 'interest52', 'interest53', 'interest54',
                'interest55', 'interest56', 'interest57', 'interest58', 'interest59', 'kw10', 'kw11', 'kw12', 'kw13',
                'kw14', 'kw15', 'kw16', 'kw17', 'kw18', 'kw19', 'kw20', 'kw21', 'kw22', 'kw23', 'kw24', 'kw25', 'kw26',
                'kw27', 'kw28', 'kw29', 'kw30', 'kw31', 'kw32', 'kw33', 'kw34', 'kw35', 'kw36', 'kw37', 'kw38', 'kw39',
                'topic10', 'topic11', 'topic12', 'topic13', 'topic14', 'topic15', 'topic16', 'topic17', 'topic18',
                'topic19', 'topic20', 'topic21', 'topic22', 'topic23', 'topic24', 'topic25', 'topic26', 'topic27',
                'topic28', 'topic29', 'topic30', 'topic31', 'topic32', 'topic33', 'topic34', 'topic35', 'topic36',
                'topic37', 'topic38', 'topic39']

    path = '../preliminary_contest_data/'  # set file path
    len_train = 8798814  # training set size
    len_test = 2265989  # test set size
    len_valid = 2000000  # set size of cross-validation set


##################################################
#          loading original data                 #
##################################################
    # extracting features from original datasets
    if not os.path.exists(path + 'processed_data.csv'):
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

        data.fillna(-1, inplace=True)
        print('extracting...')
        feature_pre_process(one_hot_features, vector_features)
        print('writing processed_data.csv...')
        data.to_csv(path + 'processed_data.csv', index=False)
    else:
        print('loading existing data...')
        data = pd.read_csv(path + 'processed_data.csv', usecols=features)


##################################################
#          feature engineering                   #
##################################################
    engineered_features = pd.DataFrame()

    print('processing app features...')
    for feature in app_features:
        mask = (data[feature] != '-1').rename(feature + '.1').apply(int)
        data = pd.concat([data, mask], axis=1)
        engineered_features[feature + '.1'] = mask
    data.drop(app_features, axis=1, inplace=True)
    del mask
    gc.collect()
    print('app features done!')

    print('adding count feature of ids...')
    for id_feature in id_features:
        idCount = data[id_feature].value_counts()
        idCount = pd.DataFrame({id_feature: idCount.index, id_feature + 'Count': idCount.values})
        data = data.merge(idCount, on=id_feature, how='left')
        engineered_features[id_feature + 'Count'] = data[id_feature + 'Count']
    del idCount
    gc.collect()
    print('count of ids done!')

    print('adding group features...')
    group_features = ['aidCount', 'uidCount', 'productType', 'age', 'gender', 'education', 'consumptionAbility',
                      'appIdInstall.1', 'carrier', 'house', 'marriageStatus.1', 'ct.1', 'os.1']
    for i in range(len(group_features) - 1):
        for j in range(i + 1, len(group_features)):
            print('grouping {} and {}'.format(group_features[i], group_features[j]))
            gp = data[[group_features[i], group_features[j], 'uid']].groupby(by=[group_features[i], group_features[j]])[['uid']].count().reset_index().rename(index=str, columns={'uid': group_features[i] + '_' + group_features[j]})
            print('merging...')
            data = data.merge(gp, on=[group_features[i], group_features[j]], how='left')
            engineered_features[group_features[i] + '_' + group_features[j]] = data[group_features[i] + '_' + group_features[j]]
            del gp
            gc.collect()

    group_user_features = ['age', 'gender', 'education', 'consumptionAbility', 'house', 'marriageStatus.1', 'os.1']
    for i in range(len(group_user_features) - 2):
        for j in range(i + 1, len(group_user_features) - 1):
            for k in range(j + 1, len(group_user_features)):
                print('grouping {}, {} and {}'.format(group_user_features[i], group_user_features[j], group_user_features[k]))
                gp = data[[group_user_features[i], group_user_features[j], group_user_features[k], 'uid']].groupby(by=[group_user_features[i], group_user_features[j], group_user_features[k]])[['uid']].count().reset_index().rename(index=str, columns={'uid': group_user_features[i] + '_' + group_user_features[j] + '_' + group_user_features[k]})
                print('merging...')
                data = data.merge(gp, on=[group_user_features[i], group_user_features[j], group_user_features[k]], how='left')
                engineered_features[group_user_features[i] + '_' + group_user_features[j] + '_' + group_user_features[k]] = data[group_user_features[i] + '_' + group_user_features[j] + '_' + group_user_features[k]]
                del gp
                gc.collect()
    print('group features done!')

    '''
    print('grouping age_gender...')
    gp = data[['age', 'gender', 'uid']].groupby(by=['age', 'gender'])[['uid']].count().reset_index().rename(index=str, columns={'uid': 'age_gender'})
    print('merging...')
    data = data.merge(gp, on=['age', 'gender'], how='left')
    del gp
    gc.collect()

    print('grouping gender_consumptionAbility...')
    gp = data[['consumptionAbility', 'gender', 'uid']].groupby(by=['consumptionAbility', 'gender'])[['uid']].count().reset_index().rename(index=str, columns={'uid': 'gender_consumptionAbility'})
    print('merging...')
    data = data.merge(gp, on=['consumptionAbility', 'gender'], how='left')
    del gp
    gc.collect()

    print('grouping age_education...')
    gp = data[['age', 'education', 'uid']].groupby(by=['age', 'education'])[['uid']].count().reset_index().rename(index=str, columns={'uid': 'age_education'})
    print('merging...')
    data = data.merge(gp, on=['age', 'education'], how='left')
    del gp
    gc.collect()

    print('grouping age_house...')
    gp = data[['age', 'house', 'uid']].groupby(by=['age', 'house'])[['uid']].count().reset_index().rename(index=str, columns={'uid': 'age_house'})
    print('merging...')
    data = data.merge(gp, on=['age', 'house'], how='left')
    del gp
    gc.collect()

    print('grouping age_gender_education_consumptionAbility...')
    gp = data[['age', 'gender', 'education', 'consumptionAbility', 'uid']].groupby(by=['age', 'gender', 'education', 'consumptionAbility'])[['uid']].count().reset_index().rename(index=str, columns={'uid': 'age_gender_edu_CA'})
    print('merging...')
    data = data.merge(gp, on=['age', 'gender', 'education', 'consumptionAbility'], how='left')
    del gp
    gc.collect()

    print('grouping age_gender_productType...')
    gp = data[['age', 'gender', 'productType', 'uid']].groupby(by=['age', 'gender', 'productType'])[['uid']].count().reset_index().rename(index=str, columns={'uid': 'age_gender_PT'})
    print('merging...')
    data = data.merge(gp, on=['age', 'gender', 'productType'], how='left')
    del gp
    gc.collect()
    '''

    print('writing engineered features to file...')
    engineered_features.to_csv(path + 'engineered_features.csv', index=False)
    del engineered_features
    gc.collect()
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
    categorical = categorical_features + id_features + ['marriageStatus.1', 'ct.1', 'os.1', 'appIdInstall.1', 'appIdAction.1']

    submission = pd.DataFrame()
    submission['aid'] = test['aid']
    submission['uid'] = test['uid']
    gc.collect()


##################################################
#          model params setting & running        #
##################################################
    print("---------------training...---------------")
    params = {
        'learning_rate': 0.05,
        # 'is_unbalance': 'true', # replaced with scale_pos_weight argument
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
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
