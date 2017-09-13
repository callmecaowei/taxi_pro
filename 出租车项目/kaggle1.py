import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection as b
from haversine import haversine
#from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import MiniBatchKMeans
import xgboost as xgb
ds=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
holidays=pd.read_csv('NYC_2016Holidays.csv')
weather=pd.read_csv('KNYC_Metars.csv')
nyc_weather=pd.read_csv('weather_data_nyc_centralpark_2016.csv')
#数据预处理阶段
nyc_weather['date']=pd.to_datetime(nyc_weather['date'])
nyc_weather.loc[nyc_weather['snow fall']=='T']=0.01
nyc_weather['year']=nyc_weather['date'].dt.year
nyc_weather['pickup_day']=nyc_weather['date'].dt.day
nyc_weather['pickup_month']=nyc_weather['date'].dt.month
nyc_weather['pickup_hour']=nyc_weather['date'].dt.hour
nyc_weather=nyc_weather[nyc_weather['year']==2016][['pickup_day','pickup_month','pickup_hour','snow fall','snow depth','maximum temerature']]
#print(nyc_weather.head())
me = np.mean(ds['trip_duration'])
st = np.std(ds['trip_duration'])
#可以理解为异常值处理
ds = ds[ds['trip_duration'] <= me + 2*st]
ds = ds[ds['trip_duration'] >= me - 2*st]
#print(ds.head(50))
ds['trip_duration_log']=(ds['trip_duration']+1).apply(np.log)
ds = ds[ds['pickup_longitude'] <= -73.75]
ds = ds[ds['pickup_longitude'] >= -74.03]
ds = ds[ds['pickup_latitude'] <= 40.85]
ds = ds[ds['pickup_latitude'] >= 40.63]
ds = ds[ds['dropoff_longitude'] <= -73.75]
ds = ds[ds['dropoff_longitude'] >= -74.03]
ds = ds[ds['dropoff_latitude'] <= 40.85]
ds = ds[ds['dropoff_latitude'] >= 40.63]

coords = np.vstack((ds[['pickup_latitude', 'pickup_longitude']].values,
                    ds[['dropoff_latitude', 'dropoff_longitude']].values))
#print(len(coords))
#由于经纬度太多太细，所以思考使用聚类降低维度，然后可以增加特征
sample_ind = np.random.permutation(len(coords))[:500000]#50w个数据
kmeans = MiniBatchKMeans(n_clusters=50, batch_size=10000).fit(coords[sample_ind])
ds.loc[:,'pickup_id']=kmeans.predict(ds[['pickup_latitude', 'pickup_longitude']])
ds.loc[:,'dropoff_id']=kmeans.predict(ds[['dropoff_latitude', 'dropoff_longitude']])
test.loc[:,'pickup_id']=kmeans.predict(test[['pickup_latitude', 'pickup_longitude']])
test.loc[:,'dropoff_id']=kmeans.predict(test[['dropoff_latitude', 'dropoff_longitude']])
ds['pickup_datetime']=pd.to_datetime(ds['pickup_datetime'])
test['pickup_datetime']=pd.to_datetime(test['pickup_datetime'])

#两个距离  ，一个大圆距离，一个测地距离
ds['distance']=ds.apply(lambda x: haversine((x['pickup_latitude'] ,x['pickup_longitude']),
                                            (x['dropoff_latitude'], x['dropoff_longitude']),miles=True),axis=1)
test['distance']=test.apply(lambda x: haversine((x['pickup_latitude'] ,x['pickup_longitude']),
                                                (x['dropoff_latitude'], x['dropoff_longitude']),miles=True),axis=1)
from geopy.distance import vincenty
ds['geo_distance']=ds.apply(lambda x: vincenty((x['pickup_latitude'] ,x['pickup_longitude']),
                                               (x['dropoff_latitude'], x['dropoff_longitude'])).miles,axis=1)
test['geo_distance']=test.apply(lambda x: vincenty((x['pickup_latitude'] ,x['pickup_longitude']),
                                                   (x['dropoff_latitude'], x['dropoff_longitude'])).miles,axis=1)
from geopy.distance import great_circle
ds['greatcircle_distance']=ds.apply(lambda x: great_circle((x['pickup_latitude'] ,x['pickup_longitude']),
                                                           (x['dropoff_latitude'], x['dropoff_longitude'])).miles,axis=1)
test['greatcircle_distance']=test.apply(lambda x: great_circle((x['pickup_latitude'] ,x['pickup_longitude']),
                                                               (x['dropoff_latitude'], x['dropoff_longitude'])).miles,axis=1)
ds['pickup_weekday']=ds['pickup_datetime'].dt.weekday
ds['pickup_hour']=ds['pickup_datetime'].dt.hour
ds['pickup_month']=ds['pickup_datetime'].dt.month
ds['pickup_day']=ds['pickup_datetime'].dt.day
test['pickup_weekday']=test['pickup_datetime'].dt.weekday
test['pickup_hour']=test['pickup_datetime'].dt.hour
test['pickup_month']=test['pickup_datetime'].dt.month
test['pickup_day']=ds['pickup_datetime'].dt.day

meanvisible=np.mean(weather['Visibility'])
weather.fillna(value=meanvisible,inplace=True)
weather['Time']=pd.to_datetime(weather['Time'])
weather['year']=weather['Time'].dt.year
weather['pickup_day']=weather['Time'].dt.day
weather['pickup_month']=weather['Time'].dt.month
weather['pickup_hour']=weather['Time'].dt.hour
weather=weather[weather['year']==2016][['pickup_day','pickup_month','pickup_hour','Temp.','Precip','Visibility']]

ds=pd.merge(ds,nyc_weather, on = ['pickup_month', 'pickup_day', 'pickup_hour'], how = 'left')
ds=pd.merge(ds,weather, on = ['pickup_month', 'pickup_day', 'pickup_hour'], how = 'left')
test=pd.merge(test,nyc_weather, on = ['pickup_month', 'pickup_day', 'pickup_hour'], how = 'left')
test=pd.merge(test,weather, on = ['pickup_month', 'pickup_day', 'pickup_hour'], how = 'left')
ds['isweekend']= ds.apply(lambda x : (x['pickup_weekday']==6 | x['pickup_weekday']==5),axis=1)
ds['isweekend']=ds['isweekend'].map({True: 1, False:0})
ds['store_and_fwd_flag']=ds['store_and_fwd_flag'].map({'N': 1, 'Y':0})
test['isweekend']= test.apply(lambda x : (x['pickup_weekday']==6 | x['pickup_weekday']==5),axis=1)
test['isweekend']=test['isweekend'].map({True: 1, False:0})
test['store_and_fwd_flag']=test['store_and_fwd_flag'].map({'N': 1, 'Y':0})
feature_cols=['vendor_id','passenger_count','pickup_id','dropoff_id','pickup_latitude','dropoff_latitude',
              'pickup_weekday','pickup_hour' ,'pickup_month','store_and_fwd_flag' ,'distance',
              'greatcircle_distance','geo_distance','Temp.','Precip','Visibility']

X=ds[feature_cols]
Y=ds['trip_duration_log']
test_features=test[feature_cols]
X_train,X_test,Y_train,Y_test= b.train_test_split(X,Y,test_size=0.2, random_state=420)
X_train,X_Val,Y_train,Y_Val= b.train_test_split(X_train,Y_train,test_size=0.1, random_state=420)
data_tr  = xgb.DMatrix(X_train, label=Y_train)
data_cv  = xgb.DMatrix(X_Val , label=Y_Val)
evallist = [(data_tr, 'train'), (data_cv, 'valid')]
parms = {'max_depth':8, #maximum depth of a tree
         'objective':'reg:linear',
         'eta'      :0.3,
         'subsample':0.9,
         'lambda'   :4,#L2正则化项
         'colsample_bytree ':0.7,
         'colsample_bylevel':1,
         'min_child_weight': 10,
         'nthread'  :-1}  #number of cpu core to use

model = xgb.train(parms, data_tr, num_boost_round=1000, evals = evallist,
                  early_stopping_rounds=30, maximize=False,
                  verbose_eval=100)

print('score = %1.5f, n_boost_round =%d.'%(model.best_score,model.best_iteration))
data_test = xgb.DMatrix(test_features)
ytest = model.predict(data_test)
xgb.plot_importance(model)
y_test = np.exp(ytest)-1
output = pd.DataFrame()
output['id'] = test['id']
output['trip_duration'] = y_test
output.to_csv('randomforest.csv', index=False)