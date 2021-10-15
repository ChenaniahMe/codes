# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 21:58:59 2020

@author: F
"""
import time
import datetime

# datetime.fromtimestamp(t) # 本地时间
# datetime.utcfromtimestamp(t) # UTC时间，世界协调时间或世界统一时间，不是北京时间
def int2time(x):
	x = int(x)
	timeArray = time.localtime(x)
	otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
	# dateArray = datetime.datetime.utcfromtimestamp(x)
	# otherStyleTime = dateArray.strftime("%Y-%m-%d %H:%M:%S")
	# strftime('%')%后跟一个控制字符，格式化datetime对象,转化成年-月-日 时：分：秒
	return otherStyleTime


def change_columns(df):
	df.columns = ['time', 'year', 'month', 'day', 'hour', 'min', 'sec', 
						'outdoorTemp', 'outdoorHum', 'outdoorAtmo', 'indoorHum', 
						'indoorAtmo', 'temperature']
	return df