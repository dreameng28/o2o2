from config import *
import pandas as pd
import matplotlib.pyplot as plt



# df = pd.read_csv(offline_train_file_path)
# df = df[df[date_received_label] != 'null']
# frame = pd.Series(list(map(lambda x, y, z: 1. if x != 'null' and y != 'null' else 0.,
#                            df[coupon_label], df[date_consumed_label], df[date_received_label])))
# frame.name = 'Label'
# df = df.join(frame)
# grouped = df[[date_received_label, frame.name]].groupby(date_received_label, as_index=False)
# print(grouped.mean())
# plt.plot(range(len(grouped.mean()[frame.name])), grouped.mean()[frame.name], marker='o')
# plt.show()

# df = pd.read_csv(online_train_file_path)
# grouped = df[[user_label, merchant_label]].groupby(merchant_label, as_index=False).count()
# grouped = grouped.sort_values(by=user_label, ascending=True)
# print(grouped)
# print(len(grouped))
# grouped = grouped[grouped[user_label] >= 100]
# print(grouped)
# print(len(grouped))

# df = pd.read_csv(offline_train_file_path)
# grouped = df[[user_label, merchant_label]].groupby(user_label, as_index=False).count()
# grouped = grouped[grouped[merchant_label] >= 50]
# users = grouped[user_label].tolist()
# grouped = df[[user_label, merchant_label]].groupby(merchant_label, as_index=False).count()
# grouped = grouped[grouped[user_label] >= 50]
# merchants = grouped[merchant_label].tolist()
# df2 = df[df[user_label].isin(users)][df[merchant_label].isin(merchants)]
# print(df)
# print(len(df))
# print(df2)
# print(len(df2))

df1 = pd.read_csv(submission_path + '_dl_2017070112PM28/submission.csv')
df2 = pd.read_csv(submission_path + '_dl_2017070112PM28/submission_xgb.csv')
print(df1)
print(df2)
series1 = df1[probability_consumed_label]
series2 = df2[probability_consumed_label]
series = pd.Series(map(lambda x, y: x/2 + y/2, series1, series2))
series.name = probability_consumed_label
df3 = df1[[user_label, coupon_label, date_received_label]].join(series)
print(df3)
df3.to_csv(submission_path + '_dl_2017070112PM28/submission.csv', index=False)
