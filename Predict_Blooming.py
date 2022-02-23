# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 12:47:38 2022

@author: annam
"""
## Prediction part from here on
    
import pandas as pd
import numpy as np
from tensorflow import keras
import os
from sklearn.preprocessing import MinMaxScaler


#%%
def make_index_date_dictionary():
    calender_dates =[]

    for i in range(1,32):
        date = str(i)+".3"
        calender_dates.append(date)
    for i in range(1,31):
        date = str(i)+".4"
        calender_dates.append(date)
    for i in range(1,32):
        date = str(i)+".5"
        calender_dates.append(date)
 
    index_date =dict(zip(range(0,92), calender_dates))
    return index_date
    
def predicted_dates(p_rediction): # Correction terms in return
    start_id = []
    full_id =[]
    scatter_id = []
    for i in range(0,len(p_rediction)):
        if p_rediction[i,1]>p_rediction[i,0] and p_rediction[i,1]>p_rediction[i,2]:
            start_id.append(i)
        if p_rediction[i,2]>p_rediction[i,1]and p_rediction[i,2]>p_rediction[i,3]:
            full_id.append(i)
    for a in range(50,92):
        if (p_rediction[a,3]>p_rediction[a,2]):
            scatter_id.append(a)
    index_date = make_index_date_dictionary()
    dates = [index_date[start_id[-1]-2], index_date[full_id[int(len(full_id)/2)]], index_date[scatter_id[0]+2]]  
    
    return dates   
    
#%% # make prediction      
    
    
def predict_dates_LRnn(X, model_temp, model_nn):
    X = np.asarray(X).astype("float32")

    prediction_LR = model_temp.predict(X).round(2) 
    prediction_LR_df = pd.DataFrame(prediction_LR)

    input_nn = pd.read_pickle("dummy_df.pkl")
    input_nn["temperature"] = prediction_LR_df[0]
    input_nn["temp_cs"]=input_nn["temperature"].cumsum() 

    input_nn=np.asarray(input_nn).astype("float32")
    input_nn =MinMaxScaler().fit_transform(input_nn) 
    prediction_LRnn = model_nn.predict(input_nn).round(2) 

    dates_LRnn =predicted_dates(prediction_LRnn) 
    list_start_full_scatter=["start", "full bloom", "scatter"]
    dates_LRnn_readable = dict(zip(list_start_full_scatter, dates_LRnn))
    print(dates_LRnn_readable)


def get_data_for_prediction(path_data, n_days):
    temp_this_year = pd.read_csv(path_data)

    temp_data=temp_this_year.iloc[59:151].reset_index(drop=True)

    temp_data[['year', 'month','day']] = temp_data['date'].str.rsplit('/', 2, expand=True)
    temp_data["temp_cs"]=temp_data["temperature"].cumsum()
    temp_data["temperature"]=temp_data["temperature"].fillna(0)
    temp_data["temp_cs"]=temp_data["temp_cs"].fillna(0)  
    temp_data.drop(["year","date"],axis='columns',inplace=True)
    temp_data.loc[n_days:,"temperature"]=0     
    temp_data.loc[n_days:,"temp_cs"]=0 
    
    return temp_data


def make_prediction(path_data, n_days):
    if n_days==10:
        model_path = os.path.abspath('GBReg_trained_10.sav')
    else:
        model_path = os.path.abspath('GB_model_30.sav')
    model_temp = pd.read_pickle(model_path)
    path_model = os.path.abspath('model_nn_trained.h5')
    model_nn = keras.models.load_model(path_model)
    temp_data = get_data_for_prediction(path_data, n_days)
    prediction = predict_dates_LRnn(temp_data, model_temp, model_nn)
    
    return prediction
#%% 


if __name__ == '__main__':
    path_data = os.path.abspath('hirosaki_this_year.csv')
    prediction = make_prediction(path_data, 10) 
    print(prediction)
