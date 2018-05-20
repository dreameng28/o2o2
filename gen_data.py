from feature_extract import *
import pandas as pd
import numpy as np


def gen_feature_data(features_dir, dataset, output):

    columns = dataset.columns.tolist()
    df = add_dataset_features(dataset)

    user_features = pd.read_csv(features_dir + 'user_features.csv', dtype=str).astype(str)
    merchant_features = pd.read_csv(features_dir + 'merchant_features.csv', dtype=str).astype(str)
    user_merchant_features = pd.read_csv(features_dir + 'user_merchant_features.csv', dtype=str).astype(str)

    df = df.merge(user_features, on=user_label, how='left')
    df = df.merge(merchant_features, on=merchant_label, how='left')
    df = df.merge(user_merchant_features, on=[user_label, merchant_label], how='left')

    df.drop(columns, axis=1, inplace=True)
    df.fillna(-1, inplace=True)
    print(df.shape)
    print('start dump feature data')
    df.to_csv(output, index=False)


def gen_label_data(dataset, output):
    df = add_label(dataset)
    print('start dump label data')
    df[['Label']].astype(float).to_csv(output, index=False)
    # print(len(df['Label'].tolist()))
    # print(df['Label'].sum()/len(df['Label'].tolist()))


def gen_data(path, label=True):
    features_dir = path + 'features/'
    dataset_file = path + 'dataset.csv'
    dataset = pd.read_csv(dataset_file, dtype=str).astype(str)
    print(dataset.shape)
    gen_feature_data(features_dir, dataset, path + 'train_features.csv')
    if label:
        gen_label_data(dataset, path + 'labels.csv')


if __name__ == '__main__':
    print('generate train data...')
    gen_data(train_path)
    print('generate validate data...')
    gen_data(validate_path)
    print('generate predict features...')
    gen_data(predict_path, label=False)
