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

    categorical = ['uid', 'age', 'gender', 'education', 'consumptionAbility', 'carrier', 'house', 'marriageStatus.1',
                   'ct.1', 'os.1', 'appIdInstall.1', 'appIdAction.1']
    results = pd.DataFrame()


##################################################
#          loading data                          #
##################################################
    path = '../preliminary_contest_data/split/aid'  # set file path
    aidList = pd.read_csv('../preliminary_contest_data/adFeature.csv', usecols=['aid'])
    aidList = aidList['aid'].tolist()
    for aid in aidList:
        print('\n********************processing aid={}********************\n'.format(aid))
        print('loading data...')
        data = pd.read_csv(path + str(aid) + '.csv')


##################################################
#          data info & training preparation      #
##################################################
        print("vars and data type: ")
        data.info()
        print(data.head(5))

        # splitting train/valid/test
        test = data[data['label'] == -1]
        len_train = len(data) - len(test)
        len_valid = len_train // 10 + 1
        train = data[:(len_train - len_valid)]
        valid = data[(len_train - len_valid):len_train]
        del data
        gc.collect()

        print("train size: ", len(train))
        print("valid size: ", len(valid))
        print("test size : ", len(test))
        print("---------------ready to run!---------------\n")

        target = 'label'
        predictors = train.columns.values.tolist()
        predictors.remove('aid')
        predictors.remove('label')

        res = pd.DataFrame()
        res['aid'] = test['aid']
        res['uid'] = test['uid']
        gc.collect()


##################################################
#          model params setting & running        #
##################################################
        print("---------------training...---------------")
        params = {
            'learning_rate': 0.001,
            # 'is_unbalance': 'true', # replaced with scale_pos_weight argument
            'num_leaves': 7,  # we should let it be smaller than 2^(max_depth)
            'max_depth': 3,  # -1 means no limit
            'min_child_samples': 10,  # Minimum number of data need in a child(min_data_in_leaf)
            'max_bin': 100,  # Number of bucketed bin for feature values
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
                                early_stopping_rounds=50,
                                verbose_eval=True,
                                num_boost_round=2000,
                                categorical_features=categorical)
        del train
        del valid
        gc.collect()

        print("---------------predicting...---------------")
        res['score'] = bst.predict(test[predictors])
        results = pd.concat([results, res])
        del test
        del res
        gc.collect()

        bst.save_model('models/model' + str(aid) + '.txt')
        # bst = lgb.Booster(model_file='model.txt')  # init model
        del bst
        gc.collect()

    print('\n********************calculating final predictions...********************')
    submission = pd.read_csv('../preliminary_contest_data/test1.csv')
    submission = submission.merge(results, on=['aid', 'uid'], how='left')
    del results
    gc.collect()
    print("---------------writing...---------------")
    submission.to_csv('submission.csv', index=False)
    print("---------------all done!---------------")
    print(submission.info())
