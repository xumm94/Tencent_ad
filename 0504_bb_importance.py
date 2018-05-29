import pandas as pd
import lightgbm as lgb
import gc


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
#          main function                         #
##################################################
if __name__ == '__main__':
    print('############################################################\n')
    print('           Tencent Algo Competition    by HBZ               \n')
    print('############################################################\n')

    features = ['aid', 'label', 'uid', 'advertiserId', 'campaignId', 'creativeSize', 'adCategoryId', 'productId',
                'productType', 'age', 'gender', 'education', 'consumptionAbility', 'LBS', 'carrier', 'house',
                'marriageStatus.1', 'ct.1', 'os.1', 'interest10', 'interest11', 'interest12', 'interest13', 'interest14',
                'interest15', 'interest16', 'interest17', 'interest18', 'interest19', 'interest20', 'interest21',
                'interest22', 'interest23', 'interest24', 'interest25', 'interest26', 'interest27', 'interest28',
                'interest29', 'interest30', 'interest31', 'interest32', 'interest33', 'interest34', 'interest35',
                'interest36', 'interest37', 'interest38', 'interest39', 'interest40', 'interest41', 'interest42',
                'interest43', 'interest44', 'interest45', 'interest46', 'interest47', 'interest48', 'interest49',
                'interest50', 'interest51', 'interest52', 'interest53', 'interest54', 'interest55', 'interest56',
                'interest57', 'interest58', 'interest59', 'kw10', 'kw11', 'kw12', 'kw13', 'kw14', 'kw15', 'kw16', 'kw17',
                'kw18', 'kw19', 'kw20', 'kw21', 'kw22', 'kw23', 'kw24', 'kw25', 'kw26', 'kw27', 'kw28', 'kw29', 'kw30',
                'kw31', 'kw32', 'kw33', 'kw34', 'kw35', 'kw36', 'kw37', 'kw38', 'kw39', 'topic10', 'topic11', 'topic12',
                'topic13', 'topic14', 'topic15', 'topic16', 'topic17', 'topic18', 'topic19', 'topic20', 'topic21',
                'topic22', 'topic23', 'topic24', 'topic25', 'topic26', 'topic27', 'topic28', 'topic29', 'topic30',
                'topic31', 'topic32', 'topic33', 'topic34', 'topic35', 'topic36', 'topic37', 'topic38', 'topic39']

    important = ['aid', 'ct.1', 'campaignId', 'advertiserId', 'marriageStatus.1', 'kw21', 'kw26', 'kw25', 'kw23', 'age',
                 'kw20', 'topic24', 'kw28', 'kw22', 'kw27', 'kw24', 'topic20', 'topic27', 'kw29', 'topic23', 'topic26',
                 'LBS', 'interest12', 'interest26', 'productType_age', 'interest21', 'kw18', 'topic28', 'topic16', 'topic22',
                 'interest28', 'topic14', 'kw17', 'topic25', 'interest27', 'interest23', 'interest25', 'kw16', 'topic21',
                 'adCategoryId', 'aidCount_age', 'uidCount_age', 'education', 'interest59', 'topic10', 'kw10', 'interest11',
                 'interest29', 'kw19', 'topic29', 'kw14', 'kw12', 'interest51', 'topic11', 'interest17', 'kw13', 'interest53',
                 'topic19', 'kw11', 'topic17', 'interest20', 'topic15', 'interest55', 'kw15', 'topic13', 'interest13',
                 'interest10', 'interest24', 'uidCount_consumptionAbility', 'uidCount_gender', 'interest52', 'age_gender',
                 'age_gender_consumptionAbility', 'interest54', 'aidCount_gender', 'interest15', 'age_gender_os.1',
                 'interest56', 'topic18', 'interest16', 'interest18', 'interest57', 'interest22', 'interest14', 'topic12',
                 'aidCount_uidCount', 'aidCount_marriageStatus.1', 'interest19', 'interest50', 'productType_consumptionAbility',
                 'interest58', 'age_consumptionAbility_house', 'age_consumptionAbility_os.1', 'age_gender_education',
                 'age_gender_marriageStatus.1', 'age_education_marriageStatus.1', 'os.1', 'productId',
                 'age_consumptionAbility_marriageStatus.1', 'age_education_consumptionAbility', 'age_marriageStatus.1_os.1']

    categorical = ['campaignId', 'education', 'os.1', 'aid', 'advertiserId', 'adCategoryId', 'age', 'productId', 'marriageStatus.1', 'ct.1']

    path = '../preliminary_contest_data/'  # set file path
    len_train = 8798814  # training set size
    len_test = 2265989  # test set size
    len_valid = 1000000  # set size of cross-validation set


##################################################
#          loading data                          #
##################################################
    # loading processed data
    print('loading processed data...')
    use1 = list(set(features).intersection(set(important))) + ['uid', 'label']
    data = pd.read_csv(path + 'processed_data.csv', usecols=use1)
    # loading engineered features (count and group)
    print('loading engineered features...')
    use2 = list(set(important) - set(use1))
    fe = pd.read_csv(path + 'engineered_features.csv', usecols=use2)
    fe2 = pd.read_csv(path + 'engineered_features_2.csv')

    print('concatenating...')
    data = pd.concat([data, fe, fe2], axis=1)
    del fe
    del fe2
    gc.collect()


##################################################
#          feature engineering                   #
##################################################
    '''
    engineered_features = pd.DataFrame()

    print('adding group features...')
    group_features = ['aid', 'ct.1', 'campaignId', 'advertiserId', 'marriageStatus.1', 'age']
    for i in range(len(group_features) - 1):
        for j in range(i + 1, len(group_features)):
            print('grouping {} and {}'.format(group_features[i], group_features[j]))
            gp = data[[group_features[i], group_features[j], 'uid']].groupby(by=[group_features[i], group_features[j]])[['uid']].count().reset_index().rename(index=str, columns={'uid': group_features[i] + '_' + group_features[j]})
            print('merging...')
            data = data.merge(gp, on=[group_features[i], group_features[j]], how='left')
            engineered_features[group_features[i] + '_' + group_features[j]] = data[group_features[i] + '_' + group_features[j]]
            del gp
            gc.collect()

    for i in range(len(group_features) - 2):
        for j in range(i + 1, len(group_features) - 1):
            for k in range(j + 1, len(group_features)):
                print('grouping {}, {} and {}'.format(group_features[i], group_features[j], group_features[k]))
                gp = data[[group_features[i], group_features[j], group_features[k], 'uid']].groupby(by=[group_features[i], group_features[j], group_features[k]])[['uid']].count().reset_index().rename(index=str, columns={'uid': group_features[i] + '_' + group_features[j] + '_' + group_features[k]})
                print('merging...')
                data = data.merge(gp, on=[group_features[i], group_features[j], group_features[k]], how='left')
                engineered_features[group_features[i] + '_' + group_features[j] + '_' + group_features[k]] = data[group_features[i] + '_' + group_features[j] + '_' + group_features[k]]
                del gp
                gc.collect()
    print('group features done!')

    print('writing engineered features to file...')
    engineered_features.to_csv(path + 'engineered_features_2.csv', index=False)
    del engineered_features
    gc.collect()
    print("---------------all features done!---------------")
    '''

##################################################
#          data info & training preparation      #
##################################################
    print("vars and data type: ")
    data.info()
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
    predictors.remove('uid')
    predictors.remove('label')

    submission = pd.DataFrame()
    submission['aid'] = test['aid']
    submission['uid'] = test['uid']
    gc.collect()


##################################################
#          model params setting & running        #
##################################################
    print("---------------training...---------------")
    params = {
        'learning_rate': 0.03,
        # 'is_unbalance': 'true', # replaced with scale_pos_weight argument
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': 5,  # -1 means no limit
        'min_child_samples': 200,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 200,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 50,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'reg_lambda': 10,  # L2 regularization term on weights
        'scale_pos_weight': 95  # because training data is extremely unbalanced
    }
    bst = lgb_modelfit_nocv(params,
                            train,
                            valid,
                            predictors,
                            target,
                            objective='binary',
                            metrics='auc',
                            early_stopping_rounds=100,
                            verbose_eval=True,
                            num_boost_round=3000,
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

    print('\nfeature importance:')
    print(bst.feature_importance())
    lgb.plot_importance(bst)
