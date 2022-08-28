# -*- coding: utf-8 -*-
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import random
import pandas as pd
# Sklearn imports
from sklearn.preprocessing import MinMaxScaler
# Tensorflow import
# import tensorflow as tf
# DiCE imports
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

np.random.seed(2333)
random.seed(2333)

output = "success"

Models = {
    'XGB': XGBRegressor(),
    'GBDT': GradientBoostingRegressor(n_estimators=1000,
                                      criterion='mse'
                                      ),
    'RF': RandomForestRegressor(),
    'MLP': MLPRegressor(),
    'DT': DecisionTreeRegressor(),
    'LR': LinearRegression(),

}

data = pd.read_csv('data.csv')  # 146 countries - 18 years (2002-2019)
data = data.drop(['iyear_country', 'Country', 'gname'], axis=1)  # delete garbage
data['nkill(mean)'] = np.log10(data['nkill(mean)'])  # log version
data['nkill(max)'] = np.log10(data['nkill(max)'])
data['nwound(mean)'] = np.log10(data['nwound(mean)'])
data['nwound(max)'] = np.log10(data['nwound(max)'])
data['count'] = np.log10(data['count'])

dataset = data.loc[np.where(data[output].notna())]
dataset = dataset[dataset[output] != 0]
dataset = dataset.reset_index(drop=True)

dataset = dataset.replace([np.inf, -np.inf], np.nan)
dataset = dataset.fillna(0)

datasetX = dataset.drop(output, axis=1)

target = dataset[output]

# # deal with categorical
# from category_encoders import TargetEncoder
# # categorical = ["Country", "attacktype", "targettype", "targetsubtype", "natlty1", "gname",
# #                "weaptype", "weapsubtype", "propextent(min)"]
# categorical = ["Country", "attacktype", "targettype", "targetsubtype", "natlty1", "gname",
#                "weaptype", "weapsubtype"]
# # numerical = datasetX.columns.difference(categorical)
# enc = TargetEncoder(cols = categorical, min_samples_leaf=20, smoothing=10)
# training_set = enc.fit_transform(datasetX, target)


# Now we have a full prediction pipeline.
for i in Models.keys():
    x_train, y_train, y_test, y_test = [], [], [], []
    if i != 'MLP':
        training_set = datasetX
        x_train = training_set[training_set['year'] < 2017].drop(['year'], axis=1)
        x_test = training_set[training_set['year'] >= 2017].drop(['year'], axis=1)
        y_train = target.loc[x_train.index]
        y_test = target.iloc[x_test.index]

    elif i == 'MLP':
        training_set = datasetX.drop(['year'], axis=1)
        year_data = datasetX['year']
        columns_name = training_set.columns
        scaler = MinMaxScaler()
        scaler.fit(training_set)
        training_set = pd.DataFrame(scaler.transform(training_set), columns=columns_name)
        training_set = pd.concat([year_data, training_set], axis=1)
        x_train = training_set[training_set['year'] < 2017].drop(['year'], axis=1)
        x_test = training_set[training_set['year'] >= 2017].drop(['year'], axis=1)
        y_train = target.loc[x_train.index]
        y_test = target.iloc[x_test.index]

    model = Models[i]
    model.fit(x_train, y_train)

    predicted = model.predict(x_test)

    MAE = mean_absolute_error(y_test, predicted)
    MSE = mean_squared_error(y_test, predicted)
    MedAE = median_absolute_error(y_test, predicted)

    print(i, "MAE: ", round(MAE, 4), "MSE: ", round(MSE, 4), "MedAE: ", round(MedAE, 4))
best_model = "RF"
model = Models[best_model]

# model.fit(x_train, y_train)
# predicted  = pd.DataFrame()
# predicted['success'] = model.predict(x_test)
# predicted = predicted.sort_values('success')
# y_ture = y_test.sort_values(ascending=True)
# fig1, ax1 = plt.subplots(figsize=(20, 10)) 
# Num = [n for n in range(len(y_test))]
# plt.plot(Num, y_ture, color = 'b')
# plt.plot(Num, predicted, color = 'r')
# plt.xlabel('Samples in test set')
# plt.ylabel('Target')
# fig1.savefig(filename, dpi=330, bbox_inches = 'tight')


# # ### FI
# best_model = "RF"
# model = Models[best_model]
# features = x_train.columns.to_list()
# FI = pd.DataFrame()
# # FI['name'] = model._Booster.feature_names #XGB
# FI['name'] = model.feature_names_in_ #RF
# FI['feature importance'] = model.feature_importances_
# FI = FI.sort_values('feature importance', ascending=False).reset_index(drop=True)
# FI = FI[FI['feature importance']>np.mean(FI['feature importance'].values)]


# plot FI
# # fig1, ax1 = plt.subplots(figsize=(20, 10)) 
# # from matplotlib.font_manager import FontProperties
# # font = FontProperties(size=88)
# plt.subplots(figsize=(10, 18)) 
# plt.barh(FI['name'], FI['feature importance'], height=1, color='grey', 
#          edgecolor='black') #
# plt.xlabel('Feature importance') #
# # plt.ylabel('features') #
# plt.title('Feature Importances') #
# for a,b in zip( FI['feature importance'],FI['name']): #
#    # print(a,b)
#    plt.text(a-0.01, b,'%.4f'%float(a), color='white') #
# plt.gca().invert_yaxis() 
# plt.savefig("FI.png", dpi=600, bbox_inches = 'tight')
# plt.show()

### ALE
# from alibi.explainers import ALE, plot_ale
# features_ALE = list(FI['name'].iloc[:5].values) + list(FI['name'].iloc[-5:].values) 
# features_ALE = FI['name'].iloc[:10].values

# model_ale = ALE(model.predict, feature_names=features, target_names=[output])
# model_exp = model_ale.explain(x_train.values, min_bin_points = 10) #EXP:EXPLAIN
# # plot_ale(model_exp, features=['targtype1'])
# axes = plot_ale(model_exp, features=features_ALE, n_cols=2, fig_kw={'figwidth':10, 'figheight': 20})

# #### SHAP
# import shap
# xx = pd.read_csv('data.csv') #146 countries - 18 years (2002-2019)
# xx = xx.drop(['gname'], axis=1) #delete garbage
# xx = xx.loc[ np.where(xx[output].notna()) ] #
# xx = xx[ xx[output] != 0] #选择非NAN
# xx = xx.reset_index(drop=True)
# xx =  xx[xx['year']<2017]

# shap.initjs() # you need this so the plots can be displayed
# explainer = shap.Explainer(model)
# shap_values = explainer(x_train)
# from shap.plots import _waterfall, waterfall
# # https://github.com/slundberg/shap/issues/2140

# num = 570
# print(xx.iloc[num])
# print("real value: ", xx[output].iloc[num])
# _waterfall.waterfall_legacy(explainer.expected_value[0], 
#                             shap_values[num].values, 
#                             x_train.iloc[num],
#                             max_display=10)
# shap.plots.force(shap_values[num], matplotlib=True)
# # shap_data = shap_values.data[analysis]
# # shap.plots.waterfall(shap_values[1], max_display=5)
# # shap.plots.force(shap_values)
# ### shap.plots.scatter(shap_values[:,"Poverty"])
# #poverty iyear
# shap.plots.bar(shap_values, max_display = 25)
