# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 11:37:13 2022

@author: annam
"""
import pandas as pd
import numpy as np

from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import os

# trains a neural net to predict blooming dates from temperatures march to may


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

class March_to_May(pd.DataFrame):   # Cuts out the periods from March to May for years 1997 to 2019
    def __init__(self, year):
        pd.DataFrame.__init__(self)
        self.year = year
        super().__init__(pd.DataFrame(sakura[sakura["year"]==str(year)].reset_index().iloc[59:151])) # March until May

def make_dict_index_year(year_start, year_end):
    n_years = year_end - year_start
    dict_index_year=dict(zip(range(0, n_years+1), range(year_start, year_end+1)))
    return dict_index_year

def make_dict_year_index(year_start, year_end):
    n_years = year_end - year_start
    dict_index_year = dict(zip(range(year_start, year_end+1), range(0, n_years+1)))
    return dict_index_year



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

    return seasons   



#%%
def make_features_nn(seasons): 
    s_all = seasons[0]
    s_all["temp_cs"]=s_all["temperature"].cumsum() 

    for i in range(1,len(seasons)):
        s_i = seasons[i]
        s_i["temp_cs"]=s_i["temperature"].cumsum()
        s_all =s_all.append(s_i)
        
    features=s_all.drop(['index', 'date', 'year', "flower_status"],axis='columns',inplace=False).reset_index(drop=True)
    X = np.asarray(features).astype("float32")
    X =MinMaxScaler().fit_transform(X) 
    return X

def make_labels_nn(seasons):
    s_all = seasons[0]
    s_all["temp_cs"]=s_all["temperature"].cumsum() 

    for i in range(1,len(seasons)):
        s_i = seasons[i]
        s_i["temp_cs"]=s_i["temperature"].cumsum()
        s_all =s_all.append(s_i)
    labels = s_all.drop(['index', 'date', 'year', "temperature",'month','day', "temp_cs"],axis='columns',inplace=False).reset_index(drop=True) # für len = 90 output
    y = np.asarray(labels).astype("float32").ravel()
    y = pd.get_dummies(y).values
    return y



def make_train_test(features, labels, test_size):
    X = np.asarray(features).astype("float32")
    y = np.asarray(labels).astype("float32")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size=test_size)
    split_data = dict(zip(["X_train", "X_test", "y_train", "y_test"], [X_train, X_test, y_train, y_test]))
    
    return split_data
#%%

def make_model_nn():
    model_nn = keras.Sequential([
	keras.layers.Dense(210, input_shape=(4,), activation='relu'), # day, month, temp, temp_cs

	keras.layers.Dense(420, activation='relu'),

	keras.layers.Dense(420, activation='relu'),
    keras.layers.Flatten(),
	keras.layers.Dense(4, activation='sigmoid')]) # 0,1,2,3 one-hot encoded
   
    model_nn.compile(optimizer="adam", 
	          loss="categorical_crossentropy",
	          metrics=['accuracy'])
    return model_nn

def get_params():
    input_batch_size = input("Enter batch size: " )
    input_num_epochs = input("Enter number of epochs: " )
    
    return {"batch_size": int(input_batch_size), "num_epochs": int(input_num_epochs)}




def train_model_nn(model, data, batch_size, epochs):
    model.fit(data["X_train"], data["y_train"], batch_size=batch_size, epochs=epochs) 
    print(model.evaluate(data["X_test"], data["y_test"]))
    return model


def plot_confusion_matrix(cm, class_names, title):
   
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max()*9 / 10
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("cm.png")
    return figure

def show_cm(X_test, y_test):
    rounded_prediction = np.argmax(model_nn.predict(X_test).round(2), axis = 1)# 


    rounded_y_test = np.argmax(y_test, axis = 1)

    cm = confusion_matrix(y_true=rounded_y_test, y_pred = rounded_prediction) 

    cm_plot_labels = ["0", "1", "2","3"]
    
    plot_confusion_matrix(cm=cm, class_names = cm_plot_labels,title = "Confusion Matrix")

def save_net(model_nn, name):
    model_nn.save(f"{name}.h5")

#%%
if __name__ == '__main__':
    path = os.path.abspath('hirosaki_temp_cherry_bloom.csv')

    sakura = load_and_clean_data(path)

    
    start_full_scatter = make_start_full_scatter(sakura)
    seasons = make_seasons(start_full_scatter["start"], start_full_scatter["full"], start_full_scatter["scatter"], 1997,2019)    

    data = make_train_test(make_features_nn(seasons), make_labels_nn(seasons), 0.3)
    model_nn = make_model_nn()
    
    batch_size_and_num_epochs = get_params()
    batch_size = batch_size_and_num_epochs["batch_size"]
    num_epochs = batch_size_and_num_epochs["num_epochs"]
    
    trained_nn = train_model_nn(model_nn, data, batch_size, num_epochs)
    
    show_cm(data["X_test"], data["y_test"])
    
    save_y_n = input("Save neural net? Y/N: ")
    if save_y_n == "Y":
        save_net(trained_nn, "trained_nn")
        print("Neural net saved")
    else:
        print("Neural net was not saved")

   
