import os
from config import *
import time
from keras.models import Sequential
from keras.layers.core import Dense, Dropout


# 加权平均
def mtd1():
    exec_time = time.strftime("%Y%m%d%I%p%M", time.localtime())
    input_dim = len(os.listdir('data/submission'))
    submissions = os.listdir('data/submission')
    tdf = pd.DataFrame()
    for each in submissions:
        current_path = os.path.join('data/submission', each)
        file_name = os.listdir(current_path)[0]
        df = pd.read_csv('data/submission/' + each + '/' + file_name)[['Probability']]
        tdf = pd.concat([tdf, df], axis=1)
    print(tdf)
    labels = tdf.apply(lambda x: x.sum() / input_dim, axis=1)
    print(labels)
    predict_features = pd.read_csv(predict_path + 'train_features.csv').astype(float)
    frame = pd.Series(labels, index=predict_features.index)
    frame.name = probability_consumed_label
    submission = pd.read_csv(predict_path + 'dataset.csv')
    submission = submission[[user_label, coupon_label, date_received_label]].join(frame)
    submission.to_csv('{0}/{1}'.format('ronghe', 'm1_' + exec_time + '.csv'), index=False)


def get_features(path):
    submissions = os.listdir('data/submission')
    tdf = pd.DataFrame()
    rem_path = path
    for each in submissions:
        if rem_path == 'submission':
            current_path = os.path.join('data/submission', each)
            path = os.listdir(current_path)[0][:-4]
            print(path)
        df = pd.read_csv('data/submission/' + each + '/' + path + '.csv')[['Probability']]
        tdf = pd.concat([tdf, df], axis=1)
    features = tdf.values
    return features


def get_labels(path):
    submissions = os.listdir('data/submission')
    labels = pd.read_csv('data/submission/' + submissions[0] + '/' + path + '.csv')[['Label']].values.ravel()
    return labels


def dl_model(input_dim, train_features, train_labels, val_features, val_labels):
    model = Sequential()
    model.add(Dense(12, input_dim=input_dim, kernel_initializer='uniform', activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(6, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(6, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(6, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Training
    model.fit(train_features, train_labels, epochs=30, batch_size=300,
              verbose=1,
              validation_data=(val_features, val_labels),
              shuffle=True)
    return model


def mtd2():
    exec_time = time.strftime("%Y%m%d%I%p%M", time.localtime())
    train_features = get_features('train')
    train_labels = get_labels('train')
    val_features = get_features('val_check')
    val_labels = get_labels('val_check')
    pre_features = get_features('submission')

    input_dim = len(os.listdir('data/submission'))
    model = dl_model(input_dim, val_features, val_labels, train_features, train_labels)

    labels = model.predict_proba(pre_features)[:, 0]
    predict_features = pd.read_csv(predict_path + 'train_features.csv').astype(float)
    frame = pd.Series(labels, index=predict_features.index)
    frame.name = probability_consumed_label
    submission = pd.read_csv(predict_path + 'dataset.csv')
    submission = submission[[user_label, coupon_label, date_received_label]].join(frame)
    submission.to_csv('{0}/{1}'.format('ronghe', 'm2_' + exec_time + '.csv'), index=False)

    train_pred_labels = model.predict_proba(train_features)[:, 0]
    val_pred_labels = model.predict_proba(val_features)[:, 0]
    train_features = pd.read_csv(train_path + 'train_features.csv').astype(float)

    validate_features = pd.read_csv(validate_path + 'train_features.csv').astype(float)
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

    print('\nAverage auc of train matrix: ', check_average_auc(train_check_matrix, 'train'))
    print('Average auc of validate matrix', check_average_auc(val_check_matrix, 'validate'))


if __name__ == '__main__':
    # mtd1()
    mtd2()
    # get_features('submission')
