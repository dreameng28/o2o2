import pandas as pd
from collections import Counter
from config import *
import matplotlib.pyplot as plt
from time import time


class DataView:
    def __init__(self, file_path=offline_train_file_path):
        self.file_path = file_path
        df = pd.read_csv(self.file_path)
        self.data = df
        self.fields = df.columns.tolist()

    @property
    def user_list(self):
        return self.data[user_label].tolist()

    @property
    def user_set(self):
        return set(self.data[user_label].tolist())

    @property
    def merchant_list(self):
        return self.data[merchant_label].tolist()

    @property
    def merchant_set(self):
        return set(self.data[merchant_label].tolist())

    @property
    def coupon_list(self):
        print(self.data[coupon_label][self.data[coupon_label].map(lambda x: str(x) not in invalid_strs)].tolist())
        return self.data[coupon_label][self.data[coupon_label].map(lambda x: str(x) not in invalid_strs)].tolist()

    @property
    def coupon_set(self):
        return set(self.data[coupon_label][self.data[coupon_label].map(lambda x: str(x) not in invalid_strs)].tolist())

    @property
    def coupon_consumption_list(self):
        # print(self.data[discount_label].map(lambda x: ':' in str(x)))
        fullcut_list = self.data[discount_label][self.data[discount_label].map(lambda x: ':' in str(x))].tolist()
        return [x.split(':')[0] for x in fullcut_list]

    @property
    def continuous_users_diff(self):
        user = '-1'
        cnt = 0
        for idx in range(self.data.shape[0]):
            if self.data.iloc[idx][user_label] != user:
                cnt += 1
                user = self.data.iloc[idx][user_label]
        return cnt

    @property
    def received_data_distribution(self):
        print(dict(self.data.groupby(date_received_label)[date_received_label].count()))
        return dict(self.data.groupby(date_received_label)[date_received_label].count())

    def filter_by_received_time(self, start_time, end_time):
        return self.data[self.data[date_received_label].map(lambda x: True if start_time <= str(x) <= end_time else False)]


def get_time_diff(date_received, date_consumed):
    # 计算时间差
    month_diff = int(date_consumed[-4:-2]) - int(date_received[-4:-2])
    if month_diff == 0:
        return int(date_consumed[-2:]) - int(date_received[-2:])
    else:
        return int(date_consumed[-2:]) - int(date_received[-2:]) + month_diff * 30


if __name__ == '__main__':
    print('data_view')

    train_offline_data = DataView(offline_train_file_path)
    test_offline_data = DataView(offline_test_file_path)
    train_online_data = DataView(online_train_file_path)
    #
    # # user view
    # train_offline_user_list, train_offline_user_set = train_offline_data.user_list, train_offline_data.user_set
    # test_offline_user_list, test_offline_user_set = test_offline_data.user_list, test_offline_data.user_set
    # train_online_user_list, train_online_user_set = train_online_data.user_list, train_online_data.user_set
    #
    # print(len(train_offline_user_list), len(train_offline_user_set))
    # print(len(test_offline_user_list), len(test_offline_user_set))
    # print(len(train_online_user_list), len(train_online_user_set))
    # train_user_set = train_online_user_set | train_offline_user_set
    # train_active_user_set = train_online_user_set & train_offline_user_set
    # print(len(train_active_user_set), len(train_user_set))
    #
    # print(len(test_offline_user_set - train_user_set))
    # print(len(test_offline_user_set - train_active_user_set))
    # print(len(test_offline_user_set - train_offline_user_set))
    # print(test_offline_user_set - train_offline_user_set)
    #
    #
    # # merchant view
    # train_offline_merchant_list, train_offline_merchant_set = train_offline_data.merchant_list, train_offline_data.merchant_set
    # test_offline_merchant_list, test_offline_merchant_set = test_offline_data.merchant_list, test_offline_data.merchant_set
    # train_online_merchant_list, train_online_merchant_set = train_online_data.merchant_list, train_online_data.merchant_set
    #
    # print(len(train_offline_merchant_list), len(train_offline_merchant_set))
    # print(len(test_offline_merchant_list), len(test_offline_merchant_set))
    # print(len(train_online_merchant_list), len(train_online_merchant_set))
    # train_merchant_set = train_online_merchant_set | train_offline_merchant_set
    # print(len(train_online_merchant_set & train_offline_merchant_set), len(train_merchant_set))
    #
    # print(len(test_offline_merchant_set - train_merchant_set))
    # print(len(test_offline_merchant_set - train_offline_merchant_set))
    # print(test_offline_merchant_set - train_offline_merchant_set)
    #
    # coupon view
    train_offline_coupon_list = train_offline_data.coupon_list
    train_offline_coupon_set = set(train_offline_coupon_list)
    test_offline_coupon_list, test_offline_coupon_set = test_offline_data.coupon_list, test_offline_data.coupon_set
    train_online_coupon_list, train_online_coupon_set = train_online_data.coupon_list, train_online_data.coupon_set

    print(len(train_offline_coupon_list), len(train_offline_coupon_set))
    print(len(test_offline_coupon_list), len(test_offline_coupon_set))
    print(len(train_online_coupon_list), len(train_online_coupon_set))
    train_coupon_set = train_online_coupon_set | train_offline_coupon_set
    print(len(train_online_coupon_set & train_offline_coupon_set), len(train_coupon_set))
    print(train_online_coupon_set & train_offline_coupon_set)
    print(len(test_offline_coupon_set - train_coupon_set))
    print(len(test_offline_coupon_set - train_offline_coupon_set))

    # coupon consumption
    # print('coupon consumption')
    # print(len(train_offline_data.coupon_consumption_list), Counter(train_offline_data.coupon_consumption_list).items())
    # print(len(test_offline_data.coupon_consumption_list), Counter(test_offline_data.coupon_consumption_list).items())
    # print(len(train_online_data.coupon_consumption_list), Counter(train_online_data.coupon_consumption_list).items())
    #
    # figure, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 8), facecolor='w', edgecolor='r')
    # ax1.hist([int(x) for x in train_offline_data.coupon_consumption_list])
    # ax2.hist([int(x) for x in test_offline_data.coupon_consumption_list])
    # ax3.hist([int(x) for x in train_online_data.coupon_consumption_list])
    # figure.suptitle('coupon consumption statistics', fontsize=16)
    # ax1.set_title('train offline data')
    # ax2.set_title('test offline data')
    # ax3.set_title('train online data')
    # plt.savefig('coupon_consumption')
    # plt.show()

    # data_sets = [train_offline_data.data, train_online_data.data, test_offline_data.data]
    # for data in data_sets:
    #     print(sum(pd.Series(list(map(lambda x, y: True if x in invalid_strs and y not in invalid_strs else False, data[coupon_label], data[date_received_label])))))
    #     print(sum(pd.Series(list(map(lambda x, y: True if x in invalid_strs and y not in invalid_strs else False, data[coupon_label], data[discount_label])))))
    #
    # print(train_offline_data.continuous_users_diff, len(train_offline_data.user_set))
    # print(train_online_data.continuous_users_diff, len(train_online_data.user_set))
    # print(test_offline_data.continuous_users_diff, len(test_offline_data.user_set))
    # 几乎所有数据都已经是按照user id做过group的

    # train_dates = train_offline_data.received_data_distribution.items()
    # train_dates = sorted(train_dates, key=lambda x: x[0])
    # print(train_dates)
    # test_dates = test_offline_data.received_data_distribution.items()
    # test_dates = sorted(test_dates, key=lambda x: x[0])
    # print(test_dates)
    #
    # start_time = time()
    # df = train_offline_data.data
    # frame = pd.Series(list(map(lambda x, y, z: 1 if x not in invalid_strs and y not in invalid_strs and get_time_diff(z, y) <= 15 else 0, df[coupon_label], df[date_consumed_label], df[date_received_label])))
    # frame.name = 'Label'
    # df = df.join(frame)
    # print(time() - start_time)
    # grouped = df.groupby([user_label, coupon_label], as_index=False, sort=False)
    # print(time() - start_time)
    # grouped = grouped.apply(lambda x: x.sort_values(by=date_received_label).reset_index())
    # print(time() - start_time)
    # df = grouped.reset_index().drop(['index', 'level_0'], axis=1).rename(columns={'level_1': 'received_order'})
    # print(time() - start_time)
    # # df.to_csv('./data/ccf_data_revised/train_offline_add_labels.csv', index=False)
    # print(time() - start_time)
