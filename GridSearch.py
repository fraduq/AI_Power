import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

estimator = lgb.LGBMRegressor(colsample_bytree=0.8, subsample=0.9, subsample_freq=5)

param_grid = {
    'learning_rate': [0.01, 0.02, 0.05, 0.1],
    'n_estimators': [100, 200, 400, 800, 1000, 1200, 1500, 2000],
    'num_leaves':[128, 1024, 4096]
}

#fit_params = {'sample_weight':, 'early_stopping_rounds':5, 'categorical_feature':[0,1,2,3,4,5]}
#fit_params = {'early_stopping_rounds':5, 'categorical_feature':[0,1,2,3,4,5]}
fit_params = {'categorical_feature':[0,1,2,3,4,5]}

gbm = GridSearchCV(estimator, param_grid, fit_params=fit_params)

gbm.fit(X_lgb, y_lgb)

print("----------------------cv results--------------------------")
print(gbm.cv_results_)

print("----------------------------cv------------------------------")
print(gbm.cv)

print('Best parameters found by grid search are:', gbm.best_params_)

#=========================================================================
#随机敲定一组参数跑模型
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
%matplotlib inline
# create dataset for lightgbm
lgb_train = lgb.Dataset(X_lgb, y_lgb)
# specify your configurations as a dict
params = {
    'num_leaves': 128,
    'learning_rate':0.01,
    'n_estimators':800,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'weight': weights,
    'application':'regression_l1'
}

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=800,
                feature_name=['f' + str(i + 1) for i in range(X_lgb.shape[1])],
                categorical_feature=[0,1,2,3,4,5])
