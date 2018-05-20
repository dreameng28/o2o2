from config import *
import datetime
import math


def get_time_diff(date_received, date_consumed):
    # 计算时间差
    month_diff = int(date_consumed[-4:-2]) - int(date_received[-4:-2])
    if month_diff == 0:
        return int(date_consumed[-2:]) - int(date_received[-2:])
    else:
        return int(date_consumed[-2:]) - int(date_received[-2:]) + month_diff * 30


def discount_floor_partition(x):
    # 满减下限分类
    x = str(x)
    if x in invalid_strs:
        return -1
    # fixed暂记为1,其实应该也算缺失值
    elif x == 'fixed':
        return 1
    # 固定折扣率是对任意消费额度都有折扣,所以考虑2,3,4,5各加一次
    elif x.find(':') == -1:
        if 0 <= float(x) <= 1:
            return 0
        else:
            print('Discount Rate Error: %s' % x)
            raise
    discount_floor = int(x.split(':')[0])
    if 0 <= discount_floor < 50:
        return 2
    elif discount_floor < 200:
        return 3
    elif discount_floor < 500:
        return 4
    else:
        return 5


def discount_rate_calculation(x):
    # 计算折率
    x = str(x)
    if x in invalid_strs or x == 'fixed':
        return -1
    elif x.find(':') == -1:
        return float(x)
    else:
        nums = x.split(':')
        return float(nums[1]) / float(nums[0])


def min_max_normalize(df, name):
    # 归一化
    max_number = df[name].max()
    min_number = df[name].min()
    # assert max_number != min_number, 'max == min in COLUMN {0}'.format(name)
    df[name] = df[name].map(lambda x: float(x - min_number + 1) / float(max_number - min_number + 1))
    # 做简单的平滑,试试效果如何
    return df


# user features

def user_normal_consume_rate(df, online=False):
    # 用户无优惠券消费率及次数归一化
    frame = pd.Series(list(map(lambda x, y: 1. if x not in invalid_strs and y in invalid_strs else 0., df[date_consumed_label], df[coupon_label])))
    frame.name = 'user_normal_consume_rate' if not online else 'online_user_normal_consume_rate'
    normal_consume_user = df[[user_label]].join(frame)
    grouped = normal_consume_user.groupby(user_label, as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'user_normal_consume_normalized_rate' if not online else 'online_user_normal_consume_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


def user_none_consume_rate(df, online=False):
    # 用户获得优惠券但没有消费率及次数归一化
    frame = pd.Series(list(map(lambda x, y: 1. if x in invalid_strs and y not in invalid_strs else 0., df[date_consumed_label], df[coupon_label])))
    frame.name = 'user_none_consume_rate' if not online else 'online_user_none_consume_rate'
    none_consume_user = df[[user_label]].join(frame)
    grouped = none_consume_user.groupby(user_label, as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'user_none_consume_normalized_rate' if not online else 'online_user_none_consume_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


def user_coupon_consume_rate(df, online=False):
    # 用户优惠券消费率及次数归一化
    frame = pd.Series(list(map(lambda x, y: 1. if x not in invalid_strs and y not in invalid_strs else 0., df[date_consumed_label], df[coupon_label])))
    frame.name = 'user_coupon_consume_rate' if not online else 'online_user_coupon_consume_rate'
    coupon_consume_user = df[[user_label]].join(frame)
    grouped = coupon_consume_user.groupby(user_label, as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'user_coupon_consume_normalized_rate' if not online else 'online_user_coupon_consume_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


def user_consume_coupon_rate(df, online=False):
    # 用户领取优惠券后进行核销率
    frame = pd.Series(list(map(lambda x, y: 1. if x not in invalid_strs and y not in invalid_strs else 0., df[date_consumed_label], df[coupon_label])))
    frame.name = 'user_consume_coupon_counts'
    coupon_consume_user = df[[user_label]].join(frame)
    frame = df[coupon_label].map(lambda x: 1. if x not in invalid_strs else 0.)
    frame.name = 'user_received_coupon_counts'
    coupon_consume_user = coupon_consume_user.join(frame)
    grouped = coupon_consume_user.groupby(user_label, as_index=False).sum()
    frame = pd.Series(list(map(lambda x, y: -1 if x == 0 else float(y) / float(x), grouped['user_received_coupon_counts'], grouped['user_consume_coupon_counts'])))
    frame.name = 'user_consume_coupon_rate' if online else 'online_user_consume_coupon_rate'
    return grouped[[user_label]].join(frame)


user_consume_rates = [user_normal_consume_rate, user_none_consume_rate, user_coupon_consume_rate, user_consume_coupon_rate]


def user_coupon_discount_floor_50_rate(df, online=False):
    # 用户优惠券消费中满0~50折扣率及次数归一化
    frame = pd.Series(list(map(lambda x, y, z: 1. if x not in invalid_strs and y not in invalid_strs and (z == 2 or z == 0) else 0., df[date_consumed_label], df[coupon_label], df['discount_floor_partition'])))
    frame.name = 'user_coupon_discount_floor_50_rate' if not online else 'online_user_coupon_discount_floor_50_rate'
    coupon_discount_user = df[[user_label]].join(frame)
    grouped = coupon_discount_user.groupby(user_label, as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'user_coupon_discount_floor_50_normalized_rate' if not online else 'online_user_coupon_discount_floor_50_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


def user_coupon_discount_floor_200_rate(df, online=False):
    # 用户优惠券消费中满50~200折扣次数归一化
    frame = pd.Series(list(map(lambda x, y, z: 1. if x not in invalid_strs and y not in invalid_strs and (z == 3 or z == 0) else 0., df[date_consumed_label], df[coupon_label], df['discount_floor_partition'])))
    frame.name = 'user_coupon_discount_floor_200_rate' if not online else 'online_user_coupon_discount_floor_200_rate'
    coupon_discount_user = df[[user_label]].join(frame)
    grouped = coupon_discount_user.groupby(user_label, as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'user_coupon_discount_floor_200_normalized_rate' if not online else 'online_user_coupon_discount_floor_200_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


def user_coupon_discount_floor_500_rate(df, online=False):
    # 用户优惠券消费中满200~500折扣次数归一化
    frame = pd.Series(list(map(lambda x, y, z: 1. if x not in invalid_strs and y not in invalid_strs and (z == 4 or z == 0) else 0., df[date_consumed_label], df[coupon_label], df['discount_floor_partition'])))
    frame.name = 'user_coupon_discount_floor_500_rate' if not online else 'online_user_coupon_discount_floor_500_rate'
    coupon_discount_user = df[[user_label]].join(frame)
    grouped = coupon_discount_user.groupby(user_label, as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'user_coupon_discount_floor_500_normalized_rate' if not online else 'online_user_coupon_discount_floor_500_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


def user_coupon_discount_floor_others_rate(df, online=False):
    # 用户优惠券消费中其他满减次数归一化
    frame = pd.Series(list(map(lambda x, y, z: 1. if x not in invalid_strs and y not in invalid_strs and (z == 5 or z == 0) else 0., df[date_consumed_label], df[coupon_label], df['discount_floor_partition'])))
    frame.name = 'user_coupon_discount_floor_others_rate' if not online else 'online_user_coupon_discount_floor_others_rate'
    coupon_discount_user = df[[user_label]].join(frame)
    grouped = coupon_discount_user.groupby(user_label, as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'user_coupon_discount_floor_others_normalized_rate' if not online else 'online_user_coupon_discount_floor_others_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


user_coupon_discount_floor_rates = [user_coupon_discount_floor_50_rate, user_coupon_discount_floor_200_rate, user_coupon_discount_floor_500_rate, user_coupon_discount_floor_others_rate]


def user_average_discount_rate(df, online=False):
    # 用户优惠券消费平均消费折率
    mask = pd.Series(list(map(lambda x, y, z: True if x not in invalid_strs and y not in invalid_strs and z != -1 else False, df[date_consumed_label], df[coupon_label], df['discount_rate'])))
    discount_rates = df[mask][[user_label, 'discount_rate']]
    discount_rates['discount_rate'] = discount_rates['discount_rate'].astype(float)
    grouped = discount_rates.groupby(user_label, as_index=False)
    return grouped['discount_rate'].mean().rename(columns={'discount_rate': 'user_discount_average_rate' if not online else 'online_user_discount_average_rate'})


def user_direct_discount_rate(df, online=False):
    # 用户优惠券消费中直接折扣消费(非满减)率及次数归一化
    mask = pd.Series(list(map(lambda x, y, z: True if x not in invalid_strs and y not in invalid_strs and z not in invalid_strs else False, df[date_consumed_label], df[coupon_label], df[discount_label])))
    discount_users = df[mask]
    frame = discount_users[discount_label].map(lambda x: 1. if str(x).find(':') == -1 else 0.)
    frame.name = 'user_direct_discount_rate' if not online else 'online_user_direct_discount_rate'
    discounts = discount_users.join(frame)[[user_label, frame.name]]
    grouped = discounts.groupby(user_label, as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'user_direct_discount_normalized_rate' if not online else 'online_user_direct_discount_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


def user_fixed_discount_rate(df, online=False):
    # 用户优惠券消费中限时低价消费率及次数归一化
    assert online, 'no fixed offline discount '
    mask = pd.Series(list(map(lambda x, y, z: True if x not in invalid_strs and y not in invalid_strs and z not in invalid_strs else False, df[date_consumed_label], df[coupon_label], df[discount_label])))
    discount_users = df[mask]
    frame = discount_users[discount_label].map(lambda x: 1. if str(x) == 'fixed' else 0.)
    frame.name = 'user_fixed_discount_rate' if not online else 'online_user_fixed_discount_rate'
    discounts = discount_users.join(frame)[[user_label, frame.name]]
    grouped = discounts.groupby(user_label, as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'user_fixed_discount_normalized_rate' if not online else 'online_user_fixed_discount_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


def user_consume_time_rate(df, online=False):
    # 用户获得优惠券后到使用消费券之间的平均等待时间计算率
    valid_time_user = df.loc[df[date_consumed_label] not in invalid_strs].loc[df[date_received_label] not in invalid_strs]
    date_consumed = valid_time_user[date_consumed_label].map(lambda x: int(x[-4:-2]) * 30 + int(x[-2:]))
    date_received = valid_time_user[date_received_label].map(lambda x: int(x[-4:-2]) * 30 + int(x[-2:]))
    frame = pd.Series(list(map(lambda x, y: 1. - float(x - y) / 15 if x - y < 15 else 0., date_consumed, date_received)), index=valid_time_user.index)
    frame.name = 'user_consume_time_rate' if not online else 'online_user_consume_time_rate'
    valid_time_user = valid_time_user[[user_label]].join(frame)
    return valid_time_user.groupby(user_label, as_index=False)[frame.name].mean()


def user_consume_merchants(df, online=False):
    # 用户优惠券消费过的不同商家数量归一化
    mask = pd.Series(list(map(lambda x, y: True if x not in invalid_strs and y not in invalid_strs else False, df[date_consumed_label], df[coupon_label])))
    grouped = df[mask][[user_label, merchant_label]].groupby(user_label)[merchant_label].nunique().reset_index()
    return min_max_normalize(grouped, merchant_label).rename(columns={merchant_label: 'user_consume_merchants_rate' if not online else 'online_user_consume_merchants_rate'})


def user_consume_coupons(df, online=False):
    # 用户优惠券消费过的不同优惠券数量归一化
    mask = pd.Series(list(map(lambda x, y: True if x not in invalid_strs and y not in invalid_strs else False, df[date_consumed_label], df[coupon_label])))
    grouped = df[mask][[user_label, coupon_label]].groupby(user_label)[coupon_label].nunique().reset_index()
    return min_max_normalize(grouped, coupon_label).rename(columns={coupon_label: 'user_consume_coupons_rate' if not online else 'online_user_consume_coupons_rate'})


def str2time(str_time):
    d = datetime.date(int(str_time[: 4]), int(str_time[4: 6]), int(str_time[6:]))
    return d


def day_duration(sd1, sd2):
    d1 = str2time(sd1)
    d2 = str2time(sd2)
    return (d2 - d1).days


def date2day(date2):
    return day_duration('20151228', date2) % 7 + 1


def coupon_receive_day(df):
    data = list(df[date_received_label].map(lambda x: date2day(x)))
    onehot_data = [0, 0, 0, 0, 0, 0, 0]
    onehot_datas = []
    for each in data:
        each = int(each) - 1
        onehot_data[each] = 1
        onehot_datas.append(onehot_data)
        onehot_data = [0, 0, 0, 0, 0, 0, 0]
    frame = pd.DataFrame(data=onehot_datas, columns=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    df = df.join(frame)
    return df


def online_user_action_0_rate(df, online=True):
    assert online, 'No action feature in offline data'
    frame = df[action_label].map(lambda x: 1. if str(x) == '0' else 0.)
    frame.name = 'online_user_action_0_rate'
    user = df[[user_label]].join(frame)
    grouped = user.groupby(user_label, as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'online_user_action_0_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


def online_user_action_1_rate(df, online=True):
    assert online, 'No action feature in offline data'
    frame = df[action_label].map(lambda x: 1. if str(x) == '1' else 0.)
    frame.name = 'online_user_action_1_rate'
    user = df[[user_label]].join(frame)
    grouped = user.groupby(user_label, as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'online_user_action_1_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


def online_user_action_2_rate(df, online=True):
    assert online, 'No action feature in offline data'
    frame = df[action_label].map(lambda x: 1. if str(x) == '2' else 0.)
    frame.name = 'online_user_action_2_rate'
    user = df[[user_label]].join(frame)
    grouped = user.groupby(user_label, as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'online_user_action_2_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


online_user_action_types = [online_user_action_0_rate, online_user_action_1_rate, online_user_action_2_rate]


def add_user_features(df, online=False):
    user_features = []
    user_features.extend(user_consume_rates)
    if online:
        user_features.extend(online_user_action_types)
        user_features.append(user_fixed_discount_rate)
    user_features.append(user_direct_discount_rate)
    user_features.append(user_consume_time_rate)
    user_features.append(user_consume_merchants)
    user_features.append(user_consume_coupons)

    user_feature_data = df[[user_label]].drop_duplicates([user_label])

    for f in user_features:
        user_feature_data = user_feature_data.merge(f(df, online=online), on=user_label, how='left')
    user_feature_data.fillna(-1, inplace=True)

    return user_feature_data


def add_user_coupon_features(df, online=False):
    frame = df[discount_label].map(discount_floor_partition)
    frame.name = 'discount_floor_partition'
    df = df.join(frame)
    frame = df[discount_label].map(discount_rate_calculation)
    frame.name = 'discount_rate'
    df = df.join(frame)

    user_coupon_features = []
    user_coupon_features.extend(user_coupon_discount_floor_rates)
    user_coupon_features.append(user_average_discount_rate)

    user_coupon_feature_data = df[[user_label]].drop_duplicates([user_label])

    for f in user_coupon_features:
        user_coupon_feature_data = user_coupon_feature_data.merge(f(df, online=online), on=user_label, how='left')
    user_coupon_feature_data.fillna(-1, inplace=True)

    return user_coupon_feature_data

# merchant features


def merchant_normal_consume_rate(df):
    # 商家无优惠券消费率及次数归一化
    frame = pd.Series(list(map(lambda x, y: 1. if x not in invalid_strs and y in invalid_strs else 0., df[date_consumed_label], df[coupon_label])))
    frame.name = 'merchant_normal_consume_rate'
    normal_consume_merchant = df[[merchant_label]].join(frame)
    grouped = normal_consume_merchant.groupby(merchant_label, as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'merchant_normal_consume_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


def merchant_none_consume_rate(df):
    # 商家获得优惠券但没有消费率及次数归一化
    frame = pd.Series(list(map(lambda x, y: 1. if x in invalid_strs and y not in invalid_strs else 0., df[date_consumed_label], df[coupon_label])))
    frame.name = 'merchant_none_consume_rate'
    none_consume_merchant = df[[merchant_label]].join(frame)
    grouped = none_consume_merchant.groupby(merchant_label, as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'merchant_none_consume_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


def merchant_coupon_consume_rate(df):
    # 商家优惠券消费率及次数归一化
    frame = pd.Series(list(map(lambda x, y: 1. if x not in invalid_strs and y not in invalid_strs else 0., df[date_consumed_label], df[coupon_label])))
    frame.name = 'merchant_coupon_consume_rate'
    coupon_consume_merchant = df[[merchant_label]].join(frame)
    grouped = coupon_consume_merchant.groupby(merchant_label, as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'merchant_coupon_consume_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


def merchant_consume_coupon_rate(df):
    # 商家的优惠券被领取后的核销率
    frame = pd.Series(list(map(lambda x, y: 1. if x not in invalid_strs and y not in invalid_strs else 0., df[date_consumed_label], df[coupon_label])))
    frame.name = 'merchant_consume_coupon_counts'
    coupon_consume_merchant = df[[merchant_label]].join(frame)
    frame = df[coupon_label].map(lambda x: 1. if x not in invalid_strs else 0.)
    frame.name = 'merchant_received_coupon_counts'
    coupon_consume_merchant = coupon_consume_merchant.join(frame)
    grouped = coupon_consume_merchant.groupby(merchant_label, as_index=False).sum()
    frame = pd.Series(list(map(lambda x, y: -1 if x == 0 else float(y) / float(x), grouped['merchant_received_coupon_counts'], grouped['merchant_consume_coupon_counts'])))
    frame.name = 'merchant_consume_coupon_rate'
    return grouped[[merchant_label]].join(frame)


merchant_consume_rates = [merchant_normal_consume_rate, merchant_none_consume_rate, merchant_consume_coupon_rate]


def merchant_coupon_discount_floor_50_rate(df):
    # 用户正常消费中满0~50折扣率及次数归一化
    frame = pd.Series(list(map(lambda x, y, z: 1. if x not in invalid_strs and y not in invalid_strs and (z == 2 or z == 0) else 0., df[date_consumed_label], df[coupon_label], df['discount_floor_partition'])))
    frame.name = 'merchant_coupon_discount_floor_50_rate'
    coupon_discount_merchant = df[[merchant_label]].join(frame)
    grouped = coupon_discount_merchant.groupby(merchant_label, as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'merchant_coupon_discount_floor_50_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


def merchant_coupon_discount_floor_200_rate(df):
    # 用户正常消费中满50~200折扣率及次数归一化
    frame = pd.Series(list(map(lambda x, y, z: 1. if x not in invalid_strs and y not in invalid_strs and (z == 3 or z == 0) else 0., df[date_consumed_label], df[coupon_label], df['discount_floor_partition'])))
    frame.name = 'merchant_coupon_discount_floor_200_rate'
    coupon_discount_merchant = df[[merchant_label]].join(frame)
    grouped = coupon_discount_merchant.groupby(merchant_label, as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'merchant_coupon_discount_floor_200_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


def merchant_coupon_discount_floor_500_rate(df):
    # 用户正常消费中满200~500折扣率及次数归一化
    frame = pd.Series(list(map(lambda x, y, z: 1. if x not in invalid_strs and y not in invalid_strs and (z == 4 or z == 0) else 0., df[date_consumed_label], df[coupon_label], df['discount_floor_partition'])))
    frame.name = 'merchant_coupon_discount_floor_500_rate'
    coupon_discount_merchant = df[[merchant_label]].join(frame)
    grouped = coupon_discount_merchant.groupby(merchant_label, as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'merchant_coupon_discount_floor_500_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


def merchant_coupon_discount_floor_others_rate(df):
    # 用户正常消费中其他折扣率及次数归一化
    frame = pd.Series(list(map(lambda x, y, z: 1. if x not in invalid_strs and y not in invalid_strs and (z == 5 or z == 0) else 0., df[date_consumed_label], df[coupon_label], df['discount_floor_partition'])))
    frame.name = 'merchant_coupon_discount_floor_others_rate'
    coupon_discount_merchant = df[[merchant_label]].join(frame)
    grouped = coupon_discount_merchant.groupby(merchant_label, as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'merchant_coupon_discount_floor_others_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


merchant_coupon_discount_floor_rates = [merchant_coupon_discount_floor_50_rate, merchant_coupon_discount_floor_200_rate, merchant_coupon_discount_floor_500_rate, merchant_coupon_discount_floor_others_rate]


def merchant_average_discount_rate(df):
    # 商家优惠券消费平均消费折率
    mask = pd.Series(list(map(lambda x, y, z: True if x not in invalid_strs and y not in invalid_strs and z not in invalid_strs else False, df[date_consumed_label], df[coupon_label], df['discount_rate'])))
    discount_rates = df[mask][[merchant_label, 'discount_rate']]
    discount_rates['discount_rate'] = discount_rates['discount_rate'].astype(float)
    grouped = discount_rates.groupby(merchant_label, as_index=False)
    return grouped['discount_rate'].mean().rename(columns={'discount_rate': 'merchant_discount_average_rate'})


def merchant_direct_discount_rate(df):
    # 商家优惠券消费中直接折扣消费(非满减)率及次数归一化
    mask = pd.Series(list(map(lambda x, y, z: True if x not in invalid_strs and y not in invalid_strs and z not in invalid_strs else False, df[date_consumed_label], df[coupon_label], df[discount_label])))
    discount_merchants = df[mask]
    frame = discount_merchants[discount_label].map(lambda x: 1. if str(x).find(':') == -1 else 0.)
    frame.name = 'merchant_direct_discount_rate'
    discounts = discount_merchants.join(frame)[[merchant_label, frame.name]]
    grouped = discounts.groupby(merchant_label, as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'merchant_direct_discount_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


def merchant_consume_time_rate(df):
    # 商家被消费的优惠券从被用户获得到使用之间的时间计算率
    valid_time_merchant = df.loc[df[date_consumed_label] not in invalid_strs].loc[df[date_received_label] not in invalid_strs]
    date_consumed = valid_time_merchant[date_consumed_label].map(lambda x: int(x[-4:-2]) * 30 + int(x[-2:]))
    date_received = valid_time_merchant[date_received_label].map(lambda x: int(x[-4:-2]) * 30 + int(x[-2:]))
    frame = pd.Series(list(map(lambda x, y: 1. - float(x - y) / 15 if x - y < 15 else 0., date_consumed, date_received)), index=valid_time_merchant.index)
    frame.name = 'merchant_consume_time_rate'
    valid_time_merchant = valid_time_merchant[[merchant_label]].join(frame)
    return valid_time_merchant.groupby(merchant_label, as_index=False)[frame.name].mean()


def merchant_consume_users(df):
    # 消费过商家优惠券的不同用户数量归一化
    mask = pd.Series(list(map(lambda x, y: True if x not in invalid_strs and y not in invalid_strs else False, df[date_consumed_label], df[coupon_label])))
    grouped = df[mask][[user_label, merchant_label]].groupby(merchant_label)[user_label].nunique().reset_index()
    return min_max_normalize(grouped, user_label).rename(columns={user_label: 'merchant_consume_users_rate'})


def merchant_consume_coupons(df):
    # 商家被消费过的不同优惠券数量归一化
    mask = pd.Series(list(map(lambda x, y: True if x not in invalid_strs and y not in invalid_strs else False, df[date_consumed_label], df[coupon_label])))
    grouped = df[mask][[coupon_label, merchant_label]].groupby(merchant_label)[coupon_label].nunique().reset_index()
    return min_max_normalize(grouped, coupon_label).rename(columns={coupon_label: 'merchant_consume_coupons_rate'})


def add_merchant_features(df):
    merchant_features = []
    merchant_features.extend(merchant_consume_rates)
    merchant_features.append(merchant_direct_discount_rate)
    merchant_features.append(merchant_consume_time_rate)
    merchant_features.append(merchant_consume_users)
    merchant_features.append(merchant_consume_coupons)

    merchant_feature_data = df[[merchant_label]].drop_duplicates([merchant_label])

    for f in merchant_features:
        merchant_feature_data = merchant_feature_data.merge(f(df), on=merchant_label, how='left')
    merchant_feature_data.fillna(-1, inplace=True)

    return merchant_feature_data


def add_merchant_coupon_features(df):
    frame = df[discount_label].map(discount_floor_partition)
    frame.name = 'discount_floor_partition'
    df = df.join(frame)
    frame = df[discount_label].map(discount_rate_calculation)
    frame.name = 'discount_rate'
    df = df.join(frame)

    merchant_coupon_features = []
    merchant_coupon_features.extend(merchant_coupon_discount_floor_rates)
    merchant_coupon_features.append(merchant_average_discount_rate)

    merchant_coupon_feature_data = df[[merchant_label]].drop_duplicates([merchant_label])

    for f in merchant_coupon_features:
        merchant_coupon_feature_data = merchant_coupon_feature_data.merge(f(df), on=merchant_label, how='left')
    merchant_coupon_feature_data.fillna(-1, inplace=True)

    return merchant_coupon_feature_data


# user merchant features


def user_merchant_normal_consume_rate(df):
    # 用户对商家的所有消费中,普通消费率及次数归一化
    frame = pd.Series(list(map(lambda x, y: 1. if x not in invalid_strs and y in invalid_strs else 0., df[date_consumed_label], df[coupon_label])))
    frame.name = 'user_merchant_normal_consume_rate'
    normal_consume_user_merchant = df[[user_label, merchant_label]].join(frame)
    grouped = normal_consume_user_merchant.groupby([user_label, merchant_label], as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'user_merchant_normal_consume_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


def user_merchant_none_consume_rate(df):
    # 用户对商家的所有消费中,不消费率及次数归一化
    frame = pd.Series(list(map(lambda x, y: 1. if x in invalid_strs and y not in invalid_strs else 0., df[date_consumed_label], df[coupon_label])))
    frame.name = 'user_merchant_none_consume_rate'
    none_consume_user_merchant = df[[user_label, merchant_label]].join(frame)
    grouped = none_consume_user_merchant.groupby([user_label, merchant_label], as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'user_merchant_none_consume_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


def user_merchant_coupon_consume_rate(df):
    # 用户对商家的所有消费中,优惠券消费率及次数归一化
    frame = pd.Series(list(map(lambda x, y: 1. if x not in invalid_strs and y in invalid_strs else 0., df[date_consumed_label], df[coupon_label])))
    frame.name = 'user_merchant_coupon_consume_rate'
    coupon_consume_user_merchant = df[[user_label, merchant_label]].join(frame)
    grouped = coupon_consume_user_merchant.groupby([user_label, merchant_label], as_index=False)
    normalized_frame = min_max_normalize(grouped[frame.name].sum(), frame.name)[frame.name]
    normalized_frame.name = 'user_merchant_coupon_consume_normalized_rate'
    return grouped[frame.name].mean().join(normalized_frame)


def user_merchant_consume_coupon_rate(df):
    # 用户领取商家的优惠券后的核销率
    frame = pd.Series(list(map(lambda x, y: 1. if x not in invalid_strs and y not in invalid_strs else 0., df[date_consumed_label], df[coupon_label])))
    frame.name = 'user_merchant_consume_coupon_counts'
    coupon_consume_merchant = df[[user_label, merchant_label]].join(frame)
    frame = df[coupon_label].map(lambda x: 1. if x not in invalid_strs else 0.)
    frame.name = 'user_merchant_received_coupon_counts'
    coupon_consume_merchant = coupon_consume_merchant.join(frame)
    grouped = coupon_consume_merchant.groupby([user_label, merchant_label], as_index=False).sum()
    frame = pd.Series(list(map(lambda x, y: -1 if x == 0 else float(y) / float(x), grouped['user_merchant_received_coupon_counts'], grouped['user_merchant_consume_coupon_counts'])))
    frame.name = 'user_merchant_consume_coupon_rate'
    return grouped[[user_label, merchant_label]].join(frame)


user_merchant_consume_rate = [user_merchant_normal_consume_rate, user_merchant_none_consume_rate, user_merchant_coupon_consume_rate, user_merchant_consume_coupon_rate]


def user_normal_consume_merchant_rate(df):
    # 用户对每个商家的普通消费次数占用户普通消费所有商家的比重
    frame = pd.Series(list(map(lambda x, y: 1. if x not in invalid_strs and y in invalid_strs else 0., df[date_consumed_label], df[coupon_label])))
    frame.name = 'normal_consume'
    normal_consume_user_merchant = df[[user_label, merchant_label]].join(frame)
    user_dicts = dict(normal_consume_user_merchant.groupby(user_label)[frame.name].sum())
    user_merchant_dicts = dict(normal_consume_user_merchant.groupby([user_label, merchant_label])[frame.name].sum())
    unique_user_merchant = df[[user_label, merchant_label]].drop_duplicates([user_label, merchant_label])
    frame = pd.Series(list(map(lambda x, y: user_merchant_dicts[(x, y)] / user_dicts[x], unique_user_merchant[user_label], unique_user_merchant[merchant_label])))
    frame.name = 'user_normal_consume_merchant_rate'
    return unique_user_merchant.join(frame)


def user_none_consume_merchant_rate(df):
    # 用户对每个商家的不消费次数占用户不消费的所有商家的比重
    frame = pd.Series(list(map(lambda x, y: 1. if x in invalid_strs and y not in invalid_strs else 0., df[date_consumed_label], df[coupon_label])))
    frame.name = 'none_consume'
    none_consume_user_merchant = df[[user_label, merchant_label]].join(frame)
    user_dicts = dict(none_consume_user_merchant.groupby(user_label)[frame.name].sum())
    user_merchant_dicts = dict(none_consume_user_merchant.groupby([user_label, merchant_label])[frame.name].sum())
    unique_user_merchant = df[[user_label, merchant_label]].drop_duplicates([user_label, merchant_label])
    frame = pd.Series(list(map(lambda x, y: user_merchant_dicts[(x, y)] / user_dicts[x], unique_user_merchant[user_label], unique_user_merchant[merchant_label])))
    frame.name = 'user_none_consume_merchant_rate'
    return unique_user_merchant.join(frame)


def user_coupon_consume_merchant_rate(df):
    # 用户对每个商家的优惠券消费次数占用户优惠券消费所有商家的比重
    frame = pd.Series(list(map(lambda x, y: 1. if x not in invalid_strs and y not in invalid_strs else 0., df[date_consumed_label], df[coupon_label])))
    frame.name = 'coupon_consume'
    coupon_consume_user_merchant = df[[user_label, merchant_label]].join(frame)
    user_dicts = dict(coupon_consume_user_merchant.groupby(user_label)[frame.name].sum())
    user_merchant_dicts = dict(coupon_consume_user_merchant.groupby([user_label, merchant_label])[frame.name].sum())
    unique_user_merchant = df[[user_label, merchant_label]].drop_duplicates([user_label, merchant_label])
    frame = pd.Series(list(map(lambda x, y: user_merchant_dicts[(x, y)] / user_dicts[x], unique_user_merchant[user_label], unique_user_merchant[merchant_label])))
    frame.name = 'user_coupon_consume_merchant_rate'
    return unique_user_merchant.join(frame)


user_consume_merchant_rate = [user_normal_consume_merchant_rate, user_none_consume_merchant_rate, user_coupon_consume_merchant_rate]


def add_user_merchant_features(df):
    user_merchant_features = []
    user_merchant_features.extend(user_merchant_consume_rate)
    user_merchant_features.extend(user_consume_merchant_rate)
    user_merchant_feature_data = df[[user_label, merchant_label]].drop_duplicates([user_label, merchant_label])

    for f in user_merchant_features:
        user_merchant_feature_data = user_merchant_feature_data.merge(f(df), on=[user_label, merchant_label], how='left')
    user_merchant_feature_data.fillna(-1, inplace=True)

    return user_merchant_feature_data


# active user online features


def add_count_users(df, data, name, online=False):
    # 人数统计
    rlp = dict(data[user_label].value_counts())
    frame = df[user_label].map(rlp)
    frame.name = name
    df = df.join(frame)

    if online:
        max_number = frame.max()
        min_number = frame.min()
        frame = frame.map(lambda x: float(x - min_number) / float(max_number - min_number))
        frame.name = '%s_normalized' % name
        df = df.join(frame)

    return df


def calc_rate(offline_column, online_column, name):
    frame = pd.Series(list(map(lambda x, y: (float(x) / (float(x) + float(y))) if float(x) + float(y) != 0 else -1, offline_column, online_column)))
    frame.name = name
    return frame


def add_offline_online_features(offline_data, active_online_data):
    active_normal_consume = active_online_data.loc[active_online_data[date_consumed_label] not in invalid_strs].loc[active_online_data[coupon_label] in invalid_strs]
    active_none_consume = active_online_data.loc[active_online_data[date_consumed_label] in invalid_strs].loc[active_online_data[coupon_label] not in invalid_strs]
    active_coupon_consume = active_online_data.loc[active_online_data[date_consumed_label] not in invalid_strs].loc[active_online_data[coupon_label] not in invalid_strs]
    new_active_data = active_online_data[[user_label]].drop_duplicates()
    new_active_data = add_count_users(new_active_data, active_normal_consume[[user_label]], 'online_normal_consume_count', online=True)
    new_active_data = add_count_users(new_active_data, active_none_consume[[user_label]], 'online_none_consume_count', online=True)
    new_active_data = add_count_users(new_active_data, active_coupon_consume[[user_label]], 'online_coupon_consume_count', online=True)
    new_active_data = add_count_users(new_active_data, active_online_data[[user_label]], 'online_count', online=True)

    offline_normal_consume = offline_data.loc[offline_data[date_consumed_label] not in invalid_strs].loc[offline_data[coupon_label] in invalid_strs]
    offline_none_consume = offline_data.loc[offline_data[date_consumed_label] in invalid_strs].loc[offline_data[coupon_label] not in invalid_strs]
    offline_coupon_consume = offline_data.loc[offline_data[date_consumed_label] not in invalid_strs].loc[offline_data[coupon_label] not in invalid_strs]
    new_offline_data = offline_data[[user_label]].drop_duplicates()
    new_offline_data = add_count_users(new_offline_data, offline_normal_consume[[user_label]], 'offline_normal_consume_count')
    new_offline_data = add_count_users(new_offline_data, offline_none_consume[[user_label]], 'offline_none_consume_count')
    new_offline_data = add_count_users(new_offline_data, offline_coupon_consume[[user_label]], 'offline_coupon_consume_count')
    new_offline_data = add_count_users(new_offline_data, offline_data, 'offline_count')

    user_online_feature_data = new_offline_data.merge(new_active_data, on=user_label, how='left')
    user_online_feature_data.fillna(0, inplace=True)
    user_online_feature_data = user_online_feature_data.join(calc_rate(user_online_feature_data['offline_normal_consume_count'], user_online_feature_data['online_normal_consume_count'], 'offline_normal_consume_rate'))
    user_online_feature_data = user_online_feature_data.join(calc_rate(user_online_feature_data['offline_none_consume_count'], user_online_feature_data['online_none_consume_count'], 'offline_none_consume_rate'))
    user_online_feature_data = user_online_feature_data.join(calc_rate(user_online_feature_data['offline_coupon_consume_count'], user_online_feature_data['online_coupon_consume_count'], 'offline_coupon_consume_rate'))
    user_online_feature_data = user_online_feature_data.join(calc_rate(user_online_feature_data['offline_count'], user_online_feature_data['online_count'], 'offline_rate'))
    user_online_feature_data = user_online_feature_data[[user_label, 'online_normal_consume_count_normalized', 'online_none_consume_count_normalized', 'online_coupon_consume_count_normalized', 'online_count_normalized', 'offline_normal_consume_rate', 'offline_none_consume_rate', 'offline_coupon_consume_rate', 'offline_rate']]

    return user_online_feature_data


# distance features


def add_distance_rate(df):
    # 距离特征
    frame = df[distance_label].map(lambda x: float(x) / 10 if x not in invalid_strs else -1)
    frame.name = 'distance_rate'
    df = df.join(frame)
    return df


# coupon features


def coupon_type(df):
    # 优惠券类型
    frame = df[discount_label].map(lambda x: -1 if x in invalid_strs else 0 if str(x).find(':') == -1 else 1)
    frame.name = 'coupon_type'
    return frame


def coupon_discount(df):
    # 优惠券折率
    frame = df[discount_label].map(discount_rate_calculation)
    frame.name = 'coupon_discount'
    return frame


def coupon_discount_floor(df):
    # 优惠券满减下限
    frame = df[discount_label].map(discount_floor_partition)
    frame.name = 'coupon_discount_floor'
    return frame


coupon_features = [coupon_type, coupon_discount, coupon_discount_floor]


def add_coupon_features(df):
    for f in coupon_features:
        df = df.join(f(df))
    return df


# dataset features

def user_received_counts(df):
    # 用户领取的所有优惠券数目及归一化
    frame = df[coupon_label].map(lambda x: 1. if x not in invalid_strs else 0.)
    frame.name = 'user_received_counts'
    received_users = df[[user_label]].join(frame)
    grouped = received_users.groupby(user_label, as_index=False).sum()
    normalized_frame = min_max_normalize(received_users, frame.name)[frame.name]
    normalized_frame.name = 'user_received_counts_normalized'
    df = df.merge(grouped.join(normalized_frame), on=user_label, how='left')
    return df


def user_received_coupon_counts(df):
    # 用户领取的特定优惠券数目及归一化
    frame = df[coupon_label].map(lambda x: 1. if x not in invalid_strs else 0.)
    frame.name = 'user_received_coupon_counts'
    received_users = df[[user_label, coupon_label]].join(frame)
    grouped = received_users.groupby([user_label, coupon_label], as_index=False).sum()
    normalized_frame = min_max_normalize(received_users, frame.name)[frame.name]
    normalized_frame.name = 'user_received_coupon_counts_normalized'
    df = df.merge(grouped.join(normalized_frame), on=[user_label, coupon_label], how='left')
    return df


def merchant_received_counts(df):
    # 商家被领取的优惠券数目及归一化
    frame = df[coupon_label].map(lambda x: 1. if x not in invalid_strs else 0.)
    frame.name = 'merchant_received_counts'
    received_merchants = df[[merchant_label]].join(frame)
    grouped = received_merchants.groupby(merchant_label, as_index=False).sum()
    normalized_frame = min_max_normalize(received_merchants, frame.name)[frame.name]
    normalized_frame.name = 'merchant_received_counts_normalized'
    df = df.merge(grouped.join(normalized_frame), on=merchant_label, how='left')
    return df


def merchant_received_coupon_counts(df):
    # 商家被领取的特定优惠券数目及归一化
    frame = df[coupon_label].map(lambda x: 1. if x not in invalid_strs else 0.)
    frame.name = 'merchant_received_coupon_counts'
    received_merchants = df[[merchant_label, coupon_label]].join(frame)
    grouped = received_merchants.groupby([merchant_label, coupon_label], as_index=False).sum()
    normalized_frame = min_max_normalize(received_merchants, frame.name)[frame.name]
    normalized_frame.name = 'merchant_received_coupon_counts_normalized'
    df = df.merge(grouped.join(normalized_frame), on=[merchant_label, coupon_label], how='left')
    return df


def user_merchant_received_counts(df):
    # 用户领取特定商家的优惠券数目及归一化
    frame = df[coupon_label].map(lambda x: 1. if x not in invalid_strs else 0.)
    frame.name = 'user_merchant_received_counts'
    received_user_merchants = df[[user_label, merchant_label]].join(frame)
    grouped = received_user_merchants.groupby([user_label, merchant_label], as_index=False).sum()
    normalized_frame = min_max_normalize(received_user_merchants, frame.name)[frame.name]
    normalized_frame.name = 'user_merchant_received_counts_normalized'
    df = df.merge(grouped.join(normalized_frame), on=[user_label, merchant_label], how='left')
    return df


coupon_received_features = [user_received_counts, user_received_coupon_counts, merchant_received_counts, merchant_received_coupon_counts, user_merchant_received_counts]


def user_merchants(df):
    grouped = df[[user_label, merchant_label]].groupby(user_label)[merchant_label].nunique().reset_index()
    normalized = min_max_normalize(grouped, merchant_label).rename(columns={merchant_label: 'user_merchants'})
    df = df.merge(normalized, on=user_label, how='left')
    return df


def merchant_users(df):
    grouped = df[[user_label, merchant_label]].groupby(merchant_label)[user_label].nunique().reset_index()
    normalized = min_max_normalize(grouped, user_label).rename(columns={user_label: 'merchant_users'})
    df = df.merge(normalized, on=merchant_label, how='left')
    return df


user_merchant_dataset_features = [user_merchants, merchant_users]


# 优惠券特征
def coupon_():
    # undo
    pass


coupon_statistical_features = []


# 一些补充的数据集特征
def user_date_received_coupon_counts(df):
    # 用户当天领取的特定优惠券数目及归一化
    frame = df[coupon_label].map(lambda x: 1. if x not in invalid_strs else 0.)
    frame.name = 'user_received_date_coupon_counts'
    received_users = df[[user_label, coupon_label, date_received_label]].join(frame)
    grouped = received_users.groupby([user_label, coupon_label, date_received_label], as_index=False).sum()
    normalized_frame = min_max_normalize(received_users, frame.name)[frame.name]
    normalized_frame.name = 'user_received_date_coupon_counts_normalized'
    df = df.merge(grouped.join(normalized_frame), on=[user_label, coupon_label, date_received_label], how='left')
    return df


def user_date_received_merchant_counts(df):
    # 用户当天领取的特定商家优惠券数目及归一化
    frame = df[coupon_label].map(lambda x: 1. if x not in invalid_strs else 0.)
    frame.name = 'user_received_date_merchant_counts'
    received_users = df[[user_label, merchant_label, date_received_label]].join(frame)
    grouped = received_users.groupby([user_label, merchant_label, date_received_label], as_index=False).sum()
    normalized_frame = min_max_normalize(received_users, frame.name)[frame.name]
    normalized_frame.name = 'user_received_date_merchant_counts_normalized'
    df = df.merge(grouped.join(normalized_frame), on=[user_label, merchant_label, date_received_label], how='left')
    return df


def user_merchant_discount_rate_counts(df):
    # 用户当天领取的特定商家优惠券数目及归一化
    frame = df[coupon_label].map(lambda x: 1. if x not in invalid_strs else 0.)
    frame.name = 'user_discount_rate_merchant_counts'
    received_users = df[[user_label, merchant_label, discount_label]].join(frame)
    grouped = received_users.groupby([user_label, merchant_label, discount_label], as_index=False).sum()
    normalized_frame = min_max_normalize(received_users, frame.name)[frame.name]
    normalized_frame.name = 'user_discount_rate_merchant_counts_normalized'
    df = df.merge(grouped.join(normalized_frame), on=[user_label, merchant_label, discount_label], how='left')
    return df


def user_merchant_discount_rates(df):
    grouped = df[[user_label, merchant_label, discount_label]].groupby([user_label, merchant_label])[discount_label].nunique().reset_index()
    grouped = grouped.rename(columns={discount_label: 'user_merchant_rates'})
    df = df.merge(grouped, on=[user_label, merchant_label], how='left')
    return df


def user_distance_rates_coupons_counts(df):
    # 用户领取的不同距离优惠券及归一化
    frame = df[coupon_label].map(lambda x: 1. if x not in invalid_strs else 0.)
    frame.name = 'user_distance_rates_counts'
    received_users = df[[user_label, distance_label]].join(frame)
    grouped = received_users.groupby([user_label, distance_label], as_index=False).sum()
    normalized_frame = min_max_normalize(received_users, frame.name)[frame.name]
    normalized_frame.name = 'user_distance_rates_counts_normalized'
    df = df.merge(grouped.join(normalized_frame), on=[user_label, distance_label], how='left')
    return df


def merchant_distance_rates_coupons_counts(df):
    # 商家发放的不同距离优惠券及归一化
    frame = df[coupon_label].map(lambda x: 1. if x not in invalid_strs else 0.)
    frame.name = 'merchant_distance_rates_counts'
    received_users = df[[merchant_label, distance_label]].join(frame)
    grouped = received_users.groupby([merchant_label, distance_label], as_index=False).sum()
    normalized_frame = min_max_normalize(received_users, frame.name)[frame.name]
    normalized_frame.name = 'merchant_distance_rates_counts_normalized'
    df = df.merge(grouped.join(normalized_frame), on=[merchant_label, distance_label], how='left')
    return df


def user_distance_rates_coupons(df):
    grouped = df[[user_label, distance_label]].groupby(user_label)[
        distance_label].nunique().reset_index()
    grouped = grouped.rename(columns={distance_label: 'user_coupon_distances'})
    df = df.merge(grouped, on=user_label, how='left')
    return df


def merchant_distance_rates_coupons(df):
    grouped = df[[merchant_label, distance_label]].groupby(merchant_label)[
        distance_label].nunique().reset_index()
    grouped = grouped.rename(columns={distance_label: 'merchant_coupon_distances'})
    df = df.merge(grouped, on=merchant_label, how='left')
    return df


def user_coupon_dates(df):
    grouped = df[[user_label, coupon_label, date_received_label]].groupby([user_label, coupon_label])[date_received_label].nunique().reset_index()
    # normalized = min_max_normalize(grouped, merchant_label).rename(columns={merchant_label: 'user_coupon_dates'})
    grouped = grouped.rename(columns={date_received_label: 'user_coupon_dates'})
    df = df.merge(grouped, on=[user_label, coupon_label], how='left')
    return df


def user_merchant_dates(df):
    grouped = df[[user_label, merchant_label, date_received_label]].groupby([user_label, merchant_label])[date_received_label].nunique().reset_index()
    # normalized = min_max_normalize(grouped, merchant_label).rename(columns={merchant_label: 'user_coupon_dates'})
    grouped = grouped.rename(columns={date_received_label: 'user_merchant_dates'})
    df = df.merge(grouped, on=[user_label, merchant_label], how='left')
    return df


def user_discount_rate_counts(df):
    # 用户领取的不同折扣优惠券及归一化
    frame = df[coupon_label].map(lambda x: 1. if x not in invalid_strs else 0.)
    frame.name = 'user_discount_rates_counts'
    received_users = df[[user_label, discount_label]].join(frame)
    grouped = received_users.groupby([user_label, discount_label], as_index=False).sum()
    normalized_frame = min_max_normalize(received_users, frame.name)[frame.name]
    normalized_frame.name = 'user_discount_rates_counts_normalized'
    df = df.merge(grouped.join(normalized_frame), on=[user_label, discount_label], how='left')
    return df


def merchant_discount_rate_counts(df):
    # 商家发放的不同折扣优惠券及归一化
    frame = df[coupon_label].map(lambda x: 1. if x not in invalid_strs else 0.)
    frame.name = 'merchant_discount_rates_counts'
    received_users = df[[merchant_label, discount_label]].join(frame)
    grouped = received_users.groupby([merchant_label, discount_label], as_index=False).sum()
    normalized_frame = min_max_normalize(received_users, frame.name)[frame.name]
    normalized_frame.name = 'merchant_discount_rates_counts_normalized'
    df = df.merge(grouped.join(normalized_frame), on=[merchant_label, discount_label], how='left')
    return df


def user_dates_counts(df):
    # 用户领券的天数
    grouped = df[[user_label, date_received_label]].groupby(user_label)[date_received_label].nunique().reset_index()
    grouped = grouped.rename(columns={date_received_label: 'user_dates_counts'})
    df = df.merge(grouped, on=user_label, how='left')
    return df


def merchant_dates_counts(df):
    # 商家发券的天数
    grouped = df[[merchant_label, date_received_label]].groupby(merchant_label)[date_received_label].nunique().reset_index()
    grouped = grouped.rename(columns={date_received_label: 'merchant_dates_counts'})
    df = df.merge(grouped, on=merchant_label, how='left')
    return df


def coupon_dates_counts(df):
    # 优惠券发放的天数
    grouped = df[[coupon_label, date_received_label]].groupby(coupon_label)[date_received_label].nunique().reset_index()
    grouped = grouped.rename(columns={date_received_label: 'coupon_dates_counts'})
    df = df.merge(grouped, on=coupon_label, how='left')
    return df


def user_before_after_counts(df):
    frame = df[coupon_label].map(lambda x: 1. if x not in invalid_strs else 0.)
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
    frame = df[coupon_label].map(lambda x: 1. if x not in invalid_strs else 0.)
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


def user_before_after_merchant_counts(df):
    frame = df[coupon_label].map(lambda x: 1. if x not in invalid_strs else 0.)
    frame.name = 'user_date_counts'
    received_users = df[[user_label, date_received_label, merchant_label]].join(frame)
    r = received_users.drop_duplicates(subset=[user_label, date_received_label, merchant_label], keep='first',
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
        a = seg_r[(seg_r[user_label] == x[0]) & (seg_r[merchant_label] == x[2]) &
                  (seg_r[date_received_label] < x[1])]
        b = seg_r[(seg_r[user_label] == x[0]) & (seg_r[merchant_label] == x[2]) &
                  (seg_r[date_received_label] > x[1])]

        return a[frame.name].sum(), b[frame.name].sum()

    temp = r.apply(lambda x: f(x), axis=1)
    s1 = temp.map(lambda x: x[0])
    s2 = temp.map(lambda x: x[1])
    s1.name = 'user_before_merchant_counts'
    s2.name = 'user_after_merchant_counts'
    r = r.join(s1).join(s2)
    return df.merge(r, on=[user_label, date_received_label, merchant_label], how='left')


additional_features = [user_before_after_counts, user_before_after_coupon_counts, user_before_after_merchant_counts,
                       user_date_received_coupon_counts, user_date_received_merchant_counts, user_merchant_discount_rate_counts,
                       user_merchant_discount_rates, user_distance_rates_coupons_counts, merchant_distance_rates_coupons_counts,
                       user_distance_rates_coupons, merchant_distance_rates_coupons, user_coupon_dates, user_merchant_dates,
                       user_discount_rate_counts, merchant_discount_rate_counts, user_dates_counts, merchant_dates_counts,
                       coupon_dates_counts]


def add_dataset_features(df):
    dataset_features = list()
    dataset_features.extend(additional_features)
    dataset_features.append(coupon_receive_day)
    dataset_features.append(add_distance_rate)
    dataset_features.append(add_coupon_features)
    dataset_features.extend(coupon_received_features)
    dataset_features.extend(user_merchant_dataset_features)

    i = 0
    for f in dataset_features:
        i += 1
        print(i, f.__name__)
        df = f(df)

    df.fillna(-1, inplace=True)
    return df


# label
def add_label(df):
    frame = pd.Series(list(map(lambda x, y, z: 1. if x not in invalid_strs and y not in invalid_strs and get_time_diff(z, y) <= 15 else 0., df[coupon_label], df[date_consumed_label], df[date_received_label])))
    frame.name = 'Label'
    print('pos_counts: {0}, neg_counts: {1}'.format(sum(frame == 1), sum(frame == 0)))
    df = df.join(frame)
    return df


# 提取特征
def extract_features(offline_data, online_data, features_path):
    print('start extract user features')
    user_feature_data = add_user_features(offline_data)
    print('start extract user coupon features')
    user_coupon_feature_data = add_user_coupon_features(offline_data)

    user_features = user_feature_data.merge(user_coupon_feature_data, on=user_label, how='outer')

    print('start extract merchant features')
    merchant_feature_data = add_merchant_features(offline_data)
    print('start extract merchant coupon features')
    merchant_coupon_feature_data = add_merchant_coupon_features(offline_data)

    merchant_features = merchant_feature_data.merge(merchant_coupon_feature_data, on=merchant_label, how='outer')

    print('start extract user merchant features')
    user_merchant_features = add_user_merchant_features(offline_data)

    print('start extract user online features')
    online_user_feature_data = add_user_features(online_data, online=True)
    print('start extract user online coupon features')
    online_user_coupon_feature_data = add_user_coupon_features(online_data, online=True)

    print('start extract user offline-online features')
    user_offline_online_feature_data = add_offline_online_features(offline_data, online_data)

    user_features = user_features.merge(online_user_feature_data, on=user_label, how='outer')
    user_features = user_features.merge(online_user_coupon_feature_data, on=user_label, how='outer')
    user_features = user_features.merge(user_offline_online_feature_data, on=user_label, how='outer')

    print('fill nan with -1')
    user_features.fillna(-1, inplace=True)
    merchant_features.fillna(-1, inplace=True)
    user_merchant_features.fillna(-1, inplace=True)

    print('start dump feature data')
    user_features.to_csv(features_path + 'user_features.csv', index=False)
    merchant_features.to_csv(features_path + 'merchant_features.csv', index=False)
    user_merchant_features.to_csv(features_path + 'user_merchant_features.csv', index=False)


def feature_extract(raw_data_path, raw_online_data_path, features_path):
    raw_data = pd.read_csv(raw_data_path, dtype=str)
    raw_online_data = pd.read_csv(raw_online_data_path, dtype=str)
    extract_features(raw_data.astype('str'), raw_online_data.astype('str'), features_path)


if __name__ == '__main__':
    print('Train features extracting...')
    feature_extract(train_raw_data_path, train_raw_online_data_path, train_feature_data_path)

    print('Validate features extracting...')
    feature_extract(validate_raw_data_path, validate_raw_online_data_path, validate_feature_data_path)

    print('Predict features extracting...')
    feature_extract(predict_raw_data_path, predict_raw_online_data_path, predict_feature_data_path)
