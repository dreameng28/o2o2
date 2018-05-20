from config import *
import time
import math


# def user_before_after_counts(df):
#     frame = df[coupon_label].map(lambda x: 1. if x != 'null' else 0.)
#     frame.name = 'user_date_counts'
#     received_users = df[[user_label, date_received_label]].join(frame)
#     min_user = received_users[user_label].astype(int).min()
#     max_user = received_users[user_label].astype(int).max()
#     seg1 = min_user + (max_user - min_user) // 3
#     seg2 = min_user + 2 * (max_user - min_user) // 3
#     r1 = received_users[received_users[user_label].astype(int) < seg1]
#     r2 = received_users[(received_users[user_label].astype(int) < seg2) &
#                         (received_users[user_label].astype(int) >= seg1)]
#     r3 = received_users[received_users[user_label].astype(int) >= seg2]
#     num = 0
#
#     def f(x, mode=0):
#         nonlocal num
#         num += 1
#         if num % 1000 == 0:
#             print(num)
#         if int(x[0]) < seg1:
#             if mode == 0:
#                 a = r1[(r1[user_label] == x[0]) & (r1[date_received_label] < x[1])]
#             else:
#                 a = r1[(r1[user_label] == x[0]) & (r1[date_received_label] > x[1])]
#         elif int(x[0]) >= seg2:
#             if mode == 0:
#                 a = r3[(r3[user_label] == x[0]) & (r3[date_received_label] < x[1])]
#             else:
#                 a = r3[(r3[user_label] == x[0]) & (r3[date_received_label] > x[1])]
#         else:
#             if mode == 0:
#                 a = r2[(r2[user_label] == x[0]) & (r2[date_received_label] < x[1])]
#             else:
#                 a = r2[(r2[user_label] == x[0]) & (r2[date_received_label] > x[1])]
#         return a[frame.name].sum()
#     r = received_users.drop_duplicates(subset=[user_label, date_received_label], keep='first', inplace=False)
#     r = r.drop(frame.name, axis=1, inplace=False)
#     r['user_before_counts'] = r.apply(lambda x: f(x), axis=1)
#     r['user_after_counts'] = r.apply(lambda x: f(x, mode=1), axis=1)
#     return df.merge(r, on=[user_label, date_received_label], how='left')
#
#
# def user_before_after_coupon_counts(df):
#     frame = df[coupon_label].map(lambda x: 1. if x != 'null' else 0.)
#     frame.name = 'user_date_counts'
#     received_users = df[[user_label, date_received_label]].join(frame)
#     min_user = received_users[user_label].astype(int).min()
#     max_user = received_users[user_label].astype(int).max()
#     seg1 = min_user + (max_user - min_user) // 3
#     seg2 = min_user + 2 * (max_user - min_user) // 3
#     r1 = received_users[received_users[user_label].astype(int) < seg1]
#     r2 = received_users[(received_users[user_label].astype(int) < seg2) &
#                         (received_users[user_label].astype(int) >= seg1)]
#     r3 = received_users[received_users[user_label].astype(int) >= seg2]
#     num = 0
#
#     def f(x, mode=0):
#         nonlocal num
#         num += 1
#         if num % 1000 == 0:
#             print(num)
#         if int(x[0]) < seg1:
#             if mode == 0:
#                 a = r1[(r1[user_label] == x[0]) & (r1[coupon_label] == x[2]) &
#                        (r1[date_received_label] < x[1])]
#             else:
#                 a = r1[(r1[user_label] == x[0]) & (r1[coupon_label] == x[2]) &
#                        (r1[date_received_label] > x[1])]
#         elif int(x[0]) >= seg2:
#             if mode == 0:
#                 a = r3[(r3[user_label] == x[0]) & (r3[coupon_label] == x[2]) &
#                        (r3[date_received_label] < x[1])]
#             else:
#                 a = r3[(r3[user_label] == x[0]) & (r3[coupon_label] == x[2]) &
#                        (r3[date_received_label] > x[1])]
#         else:
#             if mode == 0:
#                 a = r2[(r2[user_label] == x[0]) & (r2[coupon_label] == x[2]) &
#                        (r2[date_received_label] < x[1])]
#             else:
#                 a = r2[(r2[user_label] == x[0]) & (r2[coupon_label] == x[2]) &
#                        (r2[date_received_label] > x[1])]
#         return a[frame.name].sum()
#
#     r = received_users.drop_duplicates(subset=[user_label, date_received_label], keep='first', inplace=False)
#     r = r.drop(frame.name, axis=1, inplace=False)
#     r['user_before_counts'] = r.apply(lambda x: f(x), axis=1)
#     r['user_after_counts'] = r.apply(lambda x: f(x, mode=1), axis=1)
#     return df.merge(r, on=[user_label, date_received_label], how='left')


def user_before_after_counts(df):
    frame = df[coupon_label].map(lambda x: 1. if x != 'null' else 0.)
    frame.name = 'user_date_counts'
    received_users = df[[user_label, date_received_label]].join(frame)
    r = received_users.drop_duplicates(subset=[user_label, date_received_label], keep='first', inplace=False)
    r = r.drop(frame.name, axis=1, inplace=False)
    seg_num = round(math.sqrt((len(r) + len(received_users)) / 2))
    print(seg_num)
    r_u = [0] * seg_num
    for i in range(seg_num):
        r_u[i] = received_users[received_users[user_label].astype(int) % seg_num == i]
    num = 0

    def f(x):
        nonlocal num, r_u, seg_num
        num += 1
        if num % 1000 == 0:
            print(num)
        i = int(x[0]) % seg_num
        seg_r = r_u[i]
        a = seg_r[(seg_r[user_label] == x[0]) & (seg_r[date_received_label] < x[1])]
        b = seg_r[(seg_r[user_label] == x[0]) & (seg_r[date_received_label] > x[1])]

        return a[frame.name].sum(), b[frame.name].sum()

    temp = r.apply(lambda x: f(x), axis=1)
    s1 = temp.map(lambda x: x[0])
    s2 = temp.map(lambda x: x[1])
    s1.name = 'user_before_counts'
    s2.name = 'user_after_counts'
    r = r.join(s1).join(s2)
    return df.merge(r, on=[user_label, date_received_label], how='left')


def user_before_after_coupon_counts(df):
    frame = df[coupon_label].map(lambda x: 1. if x != 'null' else 0.)
    frame.name = 'user_date_counts'
    received_users = df[[user_label, date_received_label, coupon_label]].join(frame)
    r = received_users.drop_duplicates(subset=[user_label, date_received_label, coupon_label], keep='first',
                                       inplace=False)
    r = r.drop(frame.name, axis=1, inplace=False)
    seg_num = round(math.sqrt((len(r) + len(received_users)) / 2))
    r_u = [0] * seg_num
    for i in range(seg_num):
        r_u[i] = received_users[received_users[user_label].astype(int) % seg_num == i]
    num = 0

    def f(x):
        nonlocal num, r_u, seg_num
        num += 1
        if num % 1000 == 0:
            print(num)
        i = int(x[0]) % seg_num
        seg_r = r_u[i]
        a = seg_r[(seg_r[user_label] == x[0]) & (seg_r[coupon_label] == x[2]) &
                  (seg_r[date_received_label] < x[1])]
        b = seg_r[(seg_r[user_label] == x[0]) & (seg_r[coupon_label] == x[2]) &
                  (seg_r[date_received_label] > x[1])]

        return a[frame.name].sum(), b[frame.name].sum()

    temp = r.apply(lambda x: f(x), axis=1)
    s1 = temp.map(lambda x: x[0])
    s2 = temp.map(lambda x: x[1])
    s1.name = 'user_before_coupon_counts'
    s2.name = 'user_after_coupon_counts'
    r = r.join(s1).join(s2)
    return df.merge(r, on=[user_label, date_received_label, coupon_label], how='left')

# print(1)
# df = pd.read_csv(train_path + 'train_features.csv')
# df.fillna(-1, inplace=True)
# print(df)
# df.to_csv(train_path + 'train_features.csv', index=False)
# print(2)
# df = pd.read_csv(validate_path + 'train_features.csv')
# df.fillna(-1, inplace=True)
# df.to_csv(validate_path + 'train_features.csv', index=False)
# print(3)
# df = pd.read_csv(predict_path + 'train_features.csv')
# df.fillna(-1, inplace=True)
# df.to_csv(predict_path + 'train_features.csv', index=False)
#
# df = pd.DataFrame({'a': [1, 2, 3], 'b': [5, 6, np.nan]})
# df.fillna(-1, inplace=True)
# print(df)

import numpy as np
import pandas as pd
from config import *


# def sampling(features, labels):
#     labels = pd.read_csv(train_path + 'labels.csv').astype(float)
#     features = pd.read_csv(train_path + 'train_features.csv').astype(float)
#     pos_features = features[labels['Label'] == 1].values
#     neg_features = features[labels['Label'] == 0].values
#     pos_num = len(pos_features)
#     neg_num = len(neg_features)
#     indice = np.random.permutation(neg_num)
#     neg_features = neg_features[indice][: pos_num]
#     sampling_features = np.concatenate((pos_features, neg_features), axis=0)
#     sampling_labels = np.array([1] * pos_num + [0] * pos_num)
#     print(len(sampling_features), len(sampling_labels))
#     return sampling_features, sampling_labels
#
# sampling(1, 1)

# print(df.dtypes)
# print(df.shape)
# df = df[1000:2000]
# print(df)
#
#
# f = [user_before_after_counts]
# for e in f:
#     print(e.__name__)
#     t = time.time()
#     df = e(df)
#     print(time.time() - t)
#     print(df)


# def ff(df):
#     frame = df[coupon_label].map(lambda x: 1. if x != 'null' else 0.)
#     frame.name = 'user_date_counts'
#     df = df[[user_label, date_received_label]].join(frame)
#     group = df.groupby(by=[user_label])
#
#     def f(g, x):
#         a = g[g[date_received_label] < x[1]]
#         return a[frame.name].sum()
#     for name, g in group:
#         g['user_coupon_before_counts'] = g.apply(lambda x: f(g, x), axis=1)
#
# t = time.time()
# ff(df)
# print(time.time() - t)

# df = pd.read_csv(predict_dataset_path, index_col=user_label)
# df = df.sort_index()
# print(df)
# print(df.loc[[1318, 1202]])
# df = df.set_index([date_received_label, user_label])
# print(df)
import random


def pi(e):
    n1 = 0
    n2 = 0
    pi_ = 0
    while abs(pi_ - math.pi) > e:
        x = random.uniform(-.5, .5)
        y = random.uniform(-.5, .5)
        r = math.sqrt(pow(x, 2) + pow(y, 2))
        if r < .5:
            n2 += 1
        n1 += 1
        pi_ = 4 * n2 / n1
    return pi_

print(pi(0.000000001))
