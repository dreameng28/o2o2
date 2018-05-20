from sklearn.metrics import confusion_matrix
from config import *
import numpy as np
import xgboost
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import sys
import json
import operator

save_stdout = sys.stdout


def train(param, num_round=1000, early_stopping_rounds=20):
    exec_time = time.strftime("%Y%m%d%I%p%M", time.localtime())

    os.mkdir('{0}_{1}'.format(model_path, 'xgb_' + str(exec_time)))
    os.mkdir('{0}_{1}'.format(submission_path, 'xgb_' + str(exec_time)))

    train_params = param.copy()
    train_params['num_boost_round'] = num_round
    train_params['early_stopping_rounds'] = early_stopping_rounds
    json.dump(train_params, open('{0}_{1}{2}'.format(model_path, 'xgb_' + str(exec_time), model_params), 'w+'))

    print('get training data')

    train_features = pd.read_csv(train_path + 'train_features.csv').astype(float)
    train_labels = pd.read_csv(train_path + 'labels.csv').astype(float)

    validate_features = pd.read_csv(validate_path + 'train_features.csv').astype(float)
    validate_labels = pd.read_csv(validate_path + 'labels.csv').astype(float)

    predict_features = pd.read_csv(predict_path + 'train_features.csv').astype(float)

    create_feature_map(train_features.columns.tolist(), '{0}_{1}{2}'.format(model_path, 'xgb_' + str(exec_time), model_fmap_file))

    train_matrix = xgboost.DMatrix(train_features.values, label=train_labels.values, feature_names=train_features.columns)
    val_matrix = xgboost.DMatrix(validate_features.values, label=validate_labels.values, feature_names=validate_features.columns)
    predict_matrix = xgboost.DMatrix(predict_features.values, feature_names=predict_features.columns)

    watchlist = [(train_matrix, 'train'), (val_matrix, 'eval')]

    print('model training')
    with open('{0}_{1}{2}'.format(model_path, 'xgb_' + str(exec_time), model_train_log), 'w+') as outf:
        sys.stdout = outf
        model = xgboost.train(param, train_matrix, num_boost_round=num_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds)

    sys.stdout = save_stdout
    print('model.best_score: {0}, model.best_iteration: {1}, model.best_ntree_limit: {2}'.format(model.best_score, model.best_iteration, model.best_ntree_limit))

    print('output offline model data')
    model.save_model('{0}_{1}{2}'.format(model_path, 'xgb_' + str(exec_time), model_file))
    model.dump_model('{0}_{1}{2}'.format(model_path, 'xgb_' + str(exec_time), model_dump_file))

    importance = model.get_fscore(fmap='{0}_{1}{2}'.format(model_path, 'xgb_' + str(exec_time), model_fmap_file))
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    # df['fscore'] = df['fscore'] / df['fscore'].sum()
    df.to_csv('{0}_{1}{2}'.format(model_path, 'xgb_' + str(exec_time), model_feature_importance_csv), index=False)

    xgboost.plot_importance(model)
    plt.gcf().set_size_inches(20, 16)
    plt.gcf().set_tight_layout(True)
    plt.gcf().savefig('{0}_{1}{2}'.format(model_path, 'xgb_' + str(exec_time), model_feature_importance_file))
    plt.close()

    train_pred_labels = model.predict(train_matrix, ntree_limit=model.best_ntree_limit)
    val_pred_labels = model.predict(val_matrix, ntree_limit=model.best_ntree_limit)

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

    train_check_matrix.to_csv('{0}_{1}{2}'.format(submission_path, 'xgb_' + str(exec_time), '/train.csv'), index=False)
    val_check_matrix.to_csv('{0}_{1}{2}'.format(submission_path, 'xgb_' + str(exec_time), '/val_check.csv'), index=False)

    print('Average auc of train matrix: ', check_average_auc(train_check_matrix, 'train'))
    print('Average auc of validate matrix', check_average_auc(val_check_matrix, 'validate'))

    val_coupons = val_coupons.join(val_pred_frame).join(val_pred_frame.map(lambda x: 0. if x < 0.5 else 1.).rename('map')).join(val_true_frame)
    val_coupons.to_csv('{0}_{1}{2}'.format(model_path, 'xgb_' + str(exec_time), val_diff_file), index=False)
    print(confusion_matrix(val_coupons['Label'], val_coupons['map']))

    labels = model.predict(predict_matrix, ntree_limit=model.best_ntree_limit)
    frame = pd.Series(labels, index=predict_features.index)
    frame.name = probability_consumed_label

    plt.figure()
    frame.hist(figsize=(10, 8))
    plt.title('results histogram')
    plt.xlabel('predict probability')
    plt.gcf().savefig('{0}_{1}{2}'.format(submission_path, 'xgb_' + str(exec_time), submission_hist_file))
    plt.close()

    submission = pd.read_csv(predict_path + 'dataset.csv')
    submission = submission[[user_label, coupon_label, date_received_label]].join(frame)
    submission.to_csv('{0}_{1}{2}'.format(submission_path, 'xgb_' + str(exec_time), '/' +
                                          str(check_average_auc(train_check_matrix, 'train')) + '.csv'), index=False)


a = None


def aaa(x):
    import random
    if x['Coupon_id'] in a:
        return np.array(x)
    else:
        x['Probability'] = random.random()
        return np.array(x)


def eva():
    global a
    # train_check_matrix = pd.read_csv('train.csv').astype(float)
    val_check_matrix = pd.read_csv('val.csv').astype(float)
    grouped = val_check_matrix['Coupon_id'].value_counts()
    # print(grouped)
    a = grouped[grouped < 100]
    a = list(a)
    print(a)
    # print(grouped)
    # val_check_matrix.apply(aaa, axis=1)
    val_check_matrix = val_check_matrix[val_check_matrix['Coupon_id'].isin(a)]
    # print(val_check_matrix)
    # print('Average auc of train matrix: ', check_average_auc(train_check_matrix, 'train'))
    print('Average auc of validate matrix', check_average_auc(val_check_matrix, 'validate'))


if __name__ == '__main__':
    init_param = {
        'max_depth': 6,
        'eta': 0.03,
        'silent': 0,
        'seed': 17,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'scale_pos_weight': 10,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 10,
        # 'max_delta_step': 10,
        'lambda': 1,
        'alpha': 1
    }
    train(init_param, num_round=1000, early_stopping_rounds=100)
    # eva()
