from sklearn.ensemble import RandomForestClassifier
from config import *
import time
import os
from sklearn.externals import joblib


train_features = pd.read_csv(train_path + 'train_features.csv').astype(float)
train_labels = pd.read_csv(train_path + 'labels.csv').astype(float)

validate_features = pd.read_csv(validate_path + 'train_features.csv').astype(float)
validate_labels = pd.read_csv(validate_path + 'labels.csv').astype(float)

predict_features = pd.read_csv(predict_path + 'train_features.csv').astype(float)

rf = RandomForestClassifier(n_estimators=300, max_features=.7, verbose=1, n_jobs=4, criterion="entropy",
                            max_depth=6, random_state=17, oob_score=True, max_leaf_nodes=15)
exec_time = time.strftime("%Y%m%d%I%p%M", time.localtime())
os.mkdir('{0}_{1}'.format(model_path, 'rf_' + str(exec_time)))
os.mkdir('{0}_{1}'.format(submission_path, 'rf_' + str(exec_time)))
rf.fit(train_features.values, train_labels.values.ravel())
joblib.dump(rf, '{0}_{1}{2}'.format(model_path, 'rf_' + str(exec_time), '/rf_model'))


train_pred_labels = rf.predict_proba(train_features.values)[:, 1]
val_pred_labels = rf.predict_proba(validate_features.values)[:, 1]
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

train_check_matrix.to_csv('{0}_{1}{2}'.format(submission_path, 'rf_' + str(exec_time), '/train.csv'), index=False)
val_check_matrix.to_csv('{0}_{1}{2}'.format(submission_path, 'rf_' + str(exec_time), '/val_check.csv'), index=False)

print('Average auc of train matrix: ', check_average_auc(train_check_matrix, 'train'))
print('Average auc of validate matrix', check_average_auc(val_check_matrix, 'validate'))

labels = rf.predict_proba(predict_features.values)[:, 1]
frame = pd.Series(labels, index=predict_features.index)
frame.name = probability_consumed_label
submission = pd.read_csv(predict_path + 'dataset.csv')
submission = submission[[user_label, coupon_label, date_received_label]].join(frame)
submission.to_csv('{0}_{1}{2}'.format(submission_path, 'rf_' + str(exec_time), '/' +
                                      str(check_average_auc(train_check_matrix, 'train')) + '.csv'), index=False)
