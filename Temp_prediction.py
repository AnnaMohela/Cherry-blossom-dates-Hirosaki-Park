# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 12:42:47 2022

@author: annam
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pickle
from SakuraNN_refactored import make_dict_index_year, make_dict_year_index
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

#  traines a model that predicts the temperatures until the end of may from 10 or 30 days temperatur data from march

#%%

def load_and_clean_data(path_):
    sakura = pd.read_csv(path_)   
    
    sakura["flower_status"]=sakura["flower_status"].fillna(0)
    sakura.loc[sakura["flower_status"]=="bloom", "flower_status"] = "bloom starts" 
    
    sakura.loc[sakura["flower_status"]=="bloom starts", "flower_status"] = 1
    sakura.loc[sakura["flower_status"]=="full", "flower_status"] = 2
    sakura.loc[sakura["flower_status"]=="scatter", "flower_status"] = 3


    sakura[['year', 'month','day']] = sakura['date'].str.rsplit('/', 2, expand=True)
    
    return sakura


def make_start_full_scatter(sakura):
    start= sakura[sakura["flower_status"]==1].reset_index()   

    full= sakura[sakura["flower_status"]==2].reset_index()       

    scatter= sakura[sakura["flower_status"]==3].reset_index()
    return {"start": start, "full": full, "scatter": scatter}

class March_to_May(pd.DataFrame):  
    def __init__(self, year):
        pd.DataFrame.__init__(self)
        self.year = year
        super().__init__(pd.DataFrame(sakura[sakura["year"]==str(year)].reset_index().iloc[59:151])) # März bis Ende Mai



def make_seasons(start, full, scatter, start_year, end_year):
    start =  start
    full = full
    scatter= scatter
    seasons = [] 
    dict_index_year = make_dict_index_year(start_year, end_year)
    dict_year_index = make_dict_year_index(start_year, end_year)
    n_years = end_year-start_year
    for i in range(0,n_years+1):       
        season_new = March_to_May(dict_index_year[i])
        v1 =season_new.index[season_new.date == start.iloc[i, 1]].tolist()

        b1 =season_new.index[season_new.date == full.iloc[i, 1]].tolist()
        v2 =season_new.index[season_new.date == full.iloc[i, 1]].tolist()
        b2 = season_new.index[season_new.date == scatter.iloc[i, 1]].tolist()
        v3 =season_new.index[season_new.date == scatter.iloc[i, 1]].tolist()
        
        season_new.loc[v1[0]:b1[0]-1 ,"flower_status"] =1

        
        season_new.loc[ v2[0]:b2[0]-1,"flower_status"] =2

        season_new["temp_cs"] = season_new["temperature"].cumsum()
        season_new.loc[v3[0]:,"flower_status"] =3

        seasons.append(season_new)

    dummy_df= seasons[0].drop(['index', 'date', 'year',"flower_status", "temperature", "temp_cs"],axis='columns',inplace=False).reset_index(drop=True)
    dummy_df.to_pickle("dummy_df.pkl") # used later for prediction
    
    return seasons    
    
def prepare_features_10(sakura, start_year, end_year):
    start_full_scatter = make_start_full_scatter(sakura)
    start = start_full_scatter["start"]
    full = start_full_scatter["full"]
    scatter = start_full_scatter["scatter"]
    seasons = make_seasons(start, full, scatter, start_year, end_year)
    
    features_10 =[]                   
    for i in range(0,len(seasons)):
        feature =seasons[i].drop(['index', 'date', 'year',"flower_status"],axis='columns',inplace=False).reset_index(drop=True)
        feature.loc[10:,"temperature"]=0
        feature.loc[10:,"temp_cs"]=0
        features_10.append(feature)
        
    features_10_new = features_10[0] 
    for i in range(1,len(features_10)):
        feature_i = features_10[i]
        features_10_new =features_10_new.append(feature_i)
    return features_10_new
 

def prepare_features_30(sakura, start_year, end_year):
    start_full_scatter = make_start_full_scatter(sakura)
    start = start_full_scatter["start"]
    full = start_full_scatter["full"]
    scatter = start_full_scatter["scatter"]
    seasons = make_seasons(start, full, scatter, start_year, end_year)

    
    features =[]                     
    for i in range(0,len(seasons)):
        feature =seasons[i].drop(['index', 'date', 'year',"flower_status"],axis='columns',inplace=False).reset_index(drop=True)
        feature.loc[30:,"temperature"]=0
        feature.loc[30:,"temp_cs"]=0
        features.append(feature)        

    features_new = features[0]
    for i in range(1,len(features)):
        feature_i=features[i]
        features_new =features_new.append(feature_i)
      
    return features_new    

def make_labels(sakura, start_year, end_year):
    start_full_scatter = make_start_full_scatter(sakura)
    start = start_full_scatter["start"]
    full = start_full_scatter["full"]
    scatter = start_full_scatter["scatter"]
    seasons = make_seasons(start, full, scatter, start_year, end_year)

    
    s_all = seasons[0]
    for i in range(1,len(seasons)):
        s_i = seasons[i]
        s_i["temp_cs"]=s_i["temperature"].cumsum()
        s_all =s_all.append(s_i)

    labels = s_all.drop(['index', 'date', 'year', "flower_status",'month','day', "temp_cs"],axis='columns',inplace=False).reset_index(drop=True) # für len = 90 output
    
    return labels



def make_train_test(features, labels, test_size):
    X = np.asarray(features).astype("float32")
    y = np.asarray(labels).astype("float32")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size=test_size)
    split_data = dict(zip(["X_train", "X_test", "y_train", "y_test"], [X_train, X_test, y_train, y_test]))
    
    return split_data

    

#%%    Train and save model
    
def find_best_params(split_data):
    X_train_GB = split_data["X_train"]
    y_train_GB = split_data["y_train"].ravel()
    
    search_grid={'n_estimators':[500,1000,2000],'learning_rate':[.001,0.01,.1],'max_depth':[1,2,4],'subsample':[.5,.75,1],'random_state':[1]}
    GBReg_CV=GridSearchCV(estimator=GradientBoostingRegressor(),param_grid=search_grid,scoring='neg_mean_squared_error',n_jobs=1,cv=3)

    GBReg_CV.fit(X_train_GB,y_train_GB)
    best_params = GBReg_CV.best_params_

    return best_params
    
def train_GBRegressor(split_data, params):

    GBreg_best = GradientBoostingRegressor(**params)
    
    X_train_GB = split_data["X_train"]
    y_train_GB = split_data["y_train"].ravel()
    GBreg_best.fit(X_train_GB,y_train_GB)
    
    acc=GBreg_best.score(split_data["X_test"], split_data["y_test"])
    print("Accuracy : "+str(acc))
    

    return GBreg_best    

def save_model(model_name, model):
     filename = f'{model_name}.sav'
     pickle.dump(model, open(filename, 'wb'))
    

#%%

def plot_temps(prediction, y_test):
    prediction_df = pd.DataFrame(prediction.ravel()) 

    ax = sns.scatterplot(x=range(0,92),y=y_test["temperature"],color='#e37fdc', label='observed')
    sns.scatterplot(x=range(0,92),y=prediction_df[0], label=' prediction GB_regressor')

    plt.xlabel("Days from 01.03.")
    plt.ylabel("Temperature")
    plt.title("Predicted and observed temperatures (1997) ") 
    plt.show()


#%%

if __name__ == '__main__':
    path = os.path.abspath('hirosaki_temp_cherry_bloom.csv')
    sakura = load_and_clean_data(path)
    start_full_scatter = make_start_full_scatter(sakura)
    seasons = make_seasons(start_full_scatter["start"], start_full_scatter["full"], start_full_scatter["scatter"], 1997,2019)    
    start_year = 1997
    end_year=2019
    features = prepare_features_10(sakura, start_year, end_year)
    labels = make_labels(sakura, start_year, end_year)
    split_data =make_train_test(features, labels, 0.3)
    
    redo = input("Redo GridSearch_CV? Y/N: ")
    if redo == "Y":
        params = find_best_params(split_data)
    else:
        params = {'learning_rate': 0.01, 'max_depth': 2, 'n_estimators': 2000, 'random_state': 1, 'subsample': 1}# previously found through GridSearch_CV
   
    trained_GBRegressor = train_GBRegressor(split_data, params)
    X_test_1997 = features[:92]
    y_test_1997 = labels[:92]
    plot_temps(trained_GBRegressor.predict(X_test_1997), y_test_1997)
    
    save_y_n = input("Save GB_Regressor? Y/N: ")
    if save_y_n == "Y":
        save_model("GBReg_trained_10", trained_GBRegressor)
        print("GB_Regressor saved")
    else:
        print("GB_Regressor not saved")
    
