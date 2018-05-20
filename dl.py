import matplotlib.pyplot as plt
import time
from sklearn import metrics
from feature_extract import *
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import numpy as np
import os
from sklearn import preprocessing


def auc(y_true, y_pred):
    return metrics.roc_auc_score(y_true, y_pred)


def create_feature_map(features, fmap):
    outfile = open(fmap, 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def calc_auc(df):
    coupon = df[coupon_label].iloc[0]
    y_true = df['Label'].values
    if len(np.unique(y_true)) != 2:
        auc = np.nan
    else:
        y_pred = df[probability_consumed_label].values
        auc = metrics.roc_auc_score(np.array(y_true), np.array(y_pred))
    return pd.DataFrame({coupon_label: [coupon], 'auc': [auc]})


def check_average_auc(df, name):
    grouped = df.groupby(coupon_label, as_index=False).apply(lambda x: calc_auc(x))
    grouped.to_csv('{}_coupon_auc.csv'.format(name))
    return grouped['auc'].mean(skipna=True)


def sampling():
    labels = pd.read_csv(train_path + 'labels.csv').astype(int)
    features = pd.read_csv(train_path + 'train_features.csv').astype(float)
    pos_features = features[labels['Label'] == 1].values
    neg_features = features[labels['Label'] == 0].values
    pos_num = len(pos_features)
    neg_num = len(neg_features)
    indice = np.random.permutation(neg_num)
    neg_features = neg_features[indice][: pos_num]
    sampling_features = np.concatenate((pos_features, neg_features), axis=0)
    sampling_labels = np.array([1] * pos_num + [0] * pos_num)
    return sampling_features, sampling_labels


if __name__ == '__main__':
    exec_time = time.strftime("%Y%m%d%I%p%M", time.localtime())

    os.mkdir('{0}_{1}'.format(model_path, 'dl_' + str(exec_time)))
    os.mkdir('{0}_{1}'.format(submission_path, 'dl_' + str(exec_time)))

    print('get training data')
    train_features = pd.read_csv(train_path + 'train_features.csv').astype(float)
    train_labels = pd.read_csv(train_path + 'labels.csv').astype(float)

    validate_features = pd.read_csv(validate_path + 'train_features.csv').astype(float)
    validate_labels = pd.read_csv(validate_path + 'labels.csv').astype(float)

    predict_features = pd.read_csv(predict_path + 'train_features.csv').astype(float)

    dfs = [train_features, validate_features, predict_features]
    for df in dfs:
        df = df.applymap(lambda x: np.nan if x == -1. else x)

    create_feature_map(train_features.columns.tolist(), '{0}_{1}{2}'.format(model_path, 'dl_' + str(exec_time),
                                                                            '/' + model_fmap_file))

    print('Keras Training')
    model = Sequential()
    model.add(Dense(150, input_dim=136, kernel_initializer='uniform', activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(150, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    sampling_features, sampling_labels = sampling()

    # Training
    model.fit(preprocessing.scale(sampling_features), sampling_labels, epochs=30, batch_size=200, verbose=1,
              validation_data=(preprocessing.scale(validate_features.values), validate_labels.values[:, 0]),
              shuffle=True)

    # Save the model weights to a local file
    model.save_weights('{0}_{1}{2}'.format(model_path, 'dl_' + str(exec_time), '/' + model_file), overwrite=True)
    val_labels = model.predict(preprocessing.scale(validate_features.values), batch_size=16)
    print('AUC Score:', auc(np.array(validate_labels.values[:, 0]), val_labels.T[0]))

    labels = model.predict(preprocessing.scale(predict_features.values))
    labels = labels.T[0]

    print('generate submission')
    frame = pd.Series(labels, index=predict_features.index)
    frame.name = probability_consumed_label

    plt.figure()
    frame.hist(figsize=(10, 8))
    plt.title('results histogram')
    plt.xlabel('predict probability')
    plt.gcf().savefig('{0}_{1}{2}'.format(submission_path, 'dl_' + str(exec_time), '/' + submission_hist_file))

    train_pred_labels = model.predict(preprocessing.scale(train_features.values)).T[0]
    val_pred_labels = model.predict(preprocessing.scale(validate_features.values)).T[0]
    train_pred_frame = pd.Series(train_pred_labels, index=train_features.index)
    train_pred_frame.name = probability_consumed_label
    val_pred_frame = pd.Series(val_pred_labels, index=validate_features.index)
    val_pred_frame.name = probability_consumed_label

    train_true_frame = pd.read_csv(train_path + 'labels.csv')['Label']
    val_true_frame = pd.read_csv(validate_path + 'labels.csv')['Label']
    train_coupons = pd.read_csv(train_path + 'dataset.csv')
    val_coupons = pd.read_csv(validate_path + 'dataset.csv')
    train_check_matrix = train_coupons[[coupon_label]].join(train_true_frame).join(train_pred_frame)
    val_check_matrix = val_coupons[[coupon_label]].join(val_true_frame).join(val_pred_frame)

    train_check_matrix.to_csv('{0}_{1}{2}'.format(submission_path, 'dl_' + str(exec_time), '/train.csv'), index=False)
    val_check_matrix.to_csv('{0}_{1}{2}'.format(submission_path, 'dl_' + str(exec_time), '/val_check.csv'), index=False)

    print('Average auc of train matrix: ', check_average_auc(train_check_matrix, 'train'))
    print('Average auc of validate matrix: ', check_average_auc(val_check_matrix, 'validate'))

    submission = pd.read_csv(predict_path + 'dataset.csv')
    submission = submission[[user_label, coupon_label, date_received_label]].join(frame)
    submission.to_csv('{0}_{1}{2}'.format(submission_path, 'dl_' + str(exec_time), '/' +
                                             str(check_average_auc(train_check_matrix, 'train')) + '.csv'), index=False)
