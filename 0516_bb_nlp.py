import pandas as pd
import gc

path = '../preliminary_contest_data/'  # set file path
len_train = 8798814  # training set size

'''
nlpF = ['appIdInstall_score', 'appIdAction_score', 'interest1_score', 'interest2_score', 'interest5_score', 'kw1_score', 'kw2_score', 'topic1_score', 'topic2_score']
bugF = ['appIdInstall_score.1', 'appIdAction_score.1', 'interest1_score.1', 'interest2_score.1', 'interest5_score.1', 'kw1_score.1', 'kw2_score.1', 'topic1_score.1', 'topic2_score.1']

print('loading...')
nlp = pd.read_csv(path + 'nlp_features.csv')
train = nlp[:len_train]
test = nlp[len_train:]
del nlp
gc.collect()

print('processing...')
train.drop(columns=bugF, inplace=True)
test.drop(columns=nlpF, inplace=True)

table = {}
for i in range(len(nlpF)):
    table[bugF[i]] = nlpF[i]
test.rename(index=str, columns=table, inplace=True)

nlp = pd.concat([train, test], axis=0)
del train
del test
gc.collect()

print('writing...')
nlp.to_csv(path + 'nlp_features_new.csv', index=False)
'''

print('loading...')
nlp = pd.read_csv(path + 'nlp_features.csv')
data = pd.read_csv(path + 'processed_data.csv', usecols=['aid', 'campaignId'])

print('processing...')
active = pd.DataFrame(data.groupby(['campaignId']).aid.nunique()).reset_index()
active.columns = ['campaignId', 'campaignId_active_aid']
print('merging...')
data = data.merge(active, on='campaignId', how="left")

print('concatenating...')
nlp = pd.concat([nlp, data['campaignId_active_aid']], axis=1)
print('writing...')
nlp.to_csv(path + 'nlp_features_new.csv', index=False)
