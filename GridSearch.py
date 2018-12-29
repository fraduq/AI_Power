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
