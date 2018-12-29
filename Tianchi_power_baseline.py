#coding: utf-8
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("./Tianchi_power.csv")
base_df = pd.read_csv("./Tianchi_power.csv")

#print(df.head())

#获取指定的时间和日期。
df['record_date'] = pd.to_datetime(df['record_date'])
df.loc[(df.user_id==1416) & (df.record_date>'20160731'),'power_consumption'] = 1216933.0
#print(df.head())


trend = base_df[(base_df.record_date>='2015-06-01')&(base_df.record_date<'2015-09-01')]
plt.plot(trend['record_date'], trend['power_consumption'])
plt.title('2016 power consumption')
#plt.show()


trend2 = base_df[(base_df.record_date>='2016-08-27')&(base_df.record_date<'2016-09-01')]
plt.plot(trend2['record_date'], trend2['power_consumption'])
plt.title('2017 power consumption')
#plt.show()


base_df = df[['record_date', 'power_consumption']].groupby(by='record_date').agg('sum')
base_df = base_df.reset_index()
print(base_df.head())

num1 = base_df[(base_df.record_date<='2015-06-30')&(base_df.record_date>='2015-06-24')]['power_consumption'].sum()
num1 = num1/7

num2 = base_df[(base_df.record_date<='2015-07-31')&(base_df.record_date>='2015-07-25')]['power_consumption'].sum()
num2 = num2/7

num3 = base_df[(base_df.record_date<='2015-08-31')&(base_df.record_date>='2015-08-25')]['power_consumption'].sum()
num3 = num3/7

print(num1, num2, num3)
print((num1+num2-2*num3)/2)


commit_tmp_df = base_df[(base_df.record_date>='2016-07-01') & (base_df.record_date<'2016-09-01')].copy()
commit_tmp_df['day'] = base_df['record_date'].apply(lambda x: x.day)
commit_tmp_df = commit_tmp_df[['power_consumption','day']].groupby(by='day').agg('sum')
commit_tmp_df = pd.DataFrame(commit_tmp_df.reset_index())

predict_df = pd.date_range('2016/9/1', periods=31, freq='D')
predict_df = pd.DataFrame(predict_df)
predict_df.columns = ['predict_date']
predict_df['predict_power_consumption'] = commit_tmp_df['power_consumption']/2
predict_df['predict_power_consumption'] = predict_df['predict_power_consumption'].astype('int')
predict_df['predict_power_consumption'] -= diff
predict_df = predict_df[(predict_df.predict_date<'2016-10-01')]
predict_df['predict_date'] = predict_df['predict_date'].astype(str).apply(lambda x: x.replace("-",""))
predict_df.to_csv('Tianchi_power_predict_table.csv',index=False)

df2 = df[['user_id','power_consumption']].groupby(by='user_id').agg(['mean','median','var'])

df2.reset_index(col_level=1)
df2.columns = df2.columns.get_level_values(1)
df2 = df2.reset_index()
df2.head()


df_test = base_df[(base_df.record_date>='2016-08-01')&(base_df.record_date<='2016-08-30')]
df_test['record_date'] = pd.DataFrame(df_test['record_date']+pd.Timedelta('31 days'))
df_test.head()

base_df = pd.concat([base_df, df_test]).sort_values(['record_date'])

base_df['dow'] = base_df['record_date'].apply(lambda x: x.dayofweek)
base_df['doy'] = base_df['record_date'].apply(lambda x: x.dayofyear)
base_df['day'] = base_df['record_date'].apply(lambda x: x.day)
base_df['month'] = base_df['record_date'].apply(lambda x: x.month)
base_df['year'] = base_df['record_date'].apply(lambda x: x.year)

def map_season(month):
    month_dic = {1:1, 2:1, 3:2, 4:2, 5:3, 6:3, 7:3, 8:3, 9:3, 10:4, 11:4, 12:1}
    return month_dic[month]

base_df['season'] = base_df['month'].apply(lambda x: map_season(x))

base_df.head()


base_df_stats = new_df = base_df[['power_consumption','year','month']].groupby(by=['year', 'month']).agg(['mean', 'std'])
base_df_stats.head()

base_df_stats.columns = base_df_stats.columns.droplevel(0)
base_df_stats = base_df_stats.reset_index()
base_df_stats.head()

base_df_stats['1_m_mean'] = base_df_stats['mean'].shift(1)
base_df_stats['2_m_mean'] = base_df_stats['mean'].shift(2)
base_df_stats['1_m_std'] = base_df_stats['std'].shift(1)
base_df_stats['2_m_std'] = base_df_stats['std'].shift(2)
base_df_stats.head()


data_df = pd.merge(base_df, base_df_stats[['year', 'month', '1_m_mean', '2_m_mean', '1_m_std', '2_m_std']], how='inner', on=['year', 'month'])
data_df = data_df[~pd.isnull(data_df['2_m_mean'])]


data_df.to_csv('./data_all_20170524.csv', index=False)


train_data = data_df[data_df.record_date<'2016-09-01']\
[['dow','doy','day','month','year','season','1_m_mean','2_m_mean','1_m_std','2_m_std']]

test_data = data_df[data_df.record_date>='2016-09-01']\
[['dow','doy','day','month','year','season','1_m_mean','2_m_mean','1_m_std','2_m_std']]

train_target = data_df[data_df.record_date<'2016-09-01'][['power_consumption']]


train_lgb = train_data.copy()
train_lgb[['dow','doy','day','month','year','season']] = train_lgb[['dow','doy','day','month','year','season']]\
.astype(str)
test_lgb = test_data.copy()
test_lgb[['dow','doy','day','month','year','season']] = test_lgb[['dow','doy','day','month','year','season']]\
.astype(str)


X_lgb = train_lgb.values
y_lgb = train_target.values.reshape(train_target.values.shape[0],)


print(X_lgb[0,:])
