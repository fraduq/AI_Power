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
plt.show()
