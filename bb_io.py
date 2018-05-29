import pandas as pd


def dtypes_to_csv(df, path):
    df.loc[-1] = df.dtypes
    df[-1:].to_csv(path, index=False)
    df.drop(-1, inplace=True)


def dtypes_read_csv(path):
    return pd.read_csv(path).iloc[0].to_dict()


if __name__ == '__main__':
    path = ''

    # loading processed data
    print('loading processed data for stage B...')
    dtypes = dtypes_read_csv(path + 'data_stageB_dtypes.csv')
    data = pd.read_csv(path + 'data_stageB.csv', dtype=dtypes)
    print(data.info())

    # 后续操作data即可
