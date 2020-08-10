import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import sklearn as sk
import random , calendar , datetime
from sklearn.model_selection import train_test_split



def get_compiled_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10 , activation='relu'),
        tf.keras.layers.Dense( 5, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return model


def get_random_dataset(amt:int):
    times = []
    clicked = []
    weekend = [0 , 6]
    days = []
    min_time = convertToSeconds(datetime.time(0))
    max_time = convertToSeconds(datetime.time(23))
    school_start = datetime.time(8)
    school_end = datetime.time(15)
    amt_record = 0
    while(amt_record < amt):
        day = random.randint(0 ,6)
        days.append(day)
        if(day in weekend):
            random_time = random.randint(min_time , max_time)
            times.append(convertToDateTime(random_time))
            if((random_time > convertToSeconds(datetime.time(9))) and (random_time < convertToSeconds(datetime.time(11)))):
                #user_click = int(random.expovariate(lambd))
                user_click = rand1.gauss(0 , 1)
                if(user_click >= -1 and user_click <= 1.5):
                    clicked.append(True)
                else:
                    clicked.append(False)
            else:
                #user_click = int(random.expovariate(lambd))
                user_click = rand2.gauss(0 , 1)
                if(user_click >= -1 and user_click <= 1.5):
                    clicked.append(False)
                else:
                    clicked.append(True)
        else:
            random_time = random.randint(min_time , max_time)
            times.append(convertToDateTime(random_time))
            if((random_time > convertToSeconds(datetime.time(8))) and (random_time < convertToSeconds(datetime.time(11)))):
                #user_click = int(random.expovariate(lambd))
                user_click = rand3.gauss(0 , 1)
                if(user_click >= -1 and user_click <= 1.5):
                    clicked.append(True)
                else:
                    clicked.append(False)   
            else:
                #user_click = int(random.expovariate(lambd))
                user_click = rand4.gauss(0 , 1)
                if(user_click >= -1 and user_click <= 1.5):
                    clicked.append(False)
                else:
                    clicked.append(True)   
        amt_record += 1

    return (days , times , clicked)  



def transform(dataframe):
    print(dataframe) 
    dataframe['Clicked'] = pd.Categorical(dataframe['Clicked'])
    dataframe['Clicked'] = dataframe.Clicked.cat.codes
    dataframe.Time = dataframe.Time.apply(lambda x: convertToSeconds(x))
    time_col = dataframe.Time
    timeFrame = time_col.copy()
    time_np= pd.DataFrame.to_numpy(timeFrame)
    buckets = [0 , 10800 , 21600, 32400, 43200 , 54000, 64800  ,75600 , 86400 ]
    labels = ['first_eighth' , 'second_eighth' , 'third_eighth' , 'fourth_eighth' , 'fifth_eighth' , 'sixt_eighth' , 'seventh_eighth' , 'eighth_eighth']
    labeled_time = pd.cut(time_np, right=True , bins=buckets , labels=labels)
    timeFrame = dataframe.pop('Time')
    dataframe['Time'] = labeled_time
    one_hot_times = pd.get_dummies(dataframe.Time , prefix='time')
    for columns , series  in one_hot_times.items():
        dataframe[columns] = series
    dataframe.pop('Time')
    one_hot_days = pd.get_dummies(dataframe.Days , prefix='day')
    for columns , series in one_hot_days.items():
        dataframe[columns] = series
    dataframe.pop('Days')
    target = dataframe.pop('Clicked')
    dataset = tf.data.Dataset.from_tensor_slices((dataframe.values , target))
    train_dataset = dataset.shuffle(len(dataframe)).batch(1)
    return train_dataset



def convertToDateTime(seconds:int): 
    seconds = seconds % (24 * 3600) 
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return datetime.time(hour=hour , second=seconds , minute=minutes)



def convertToSeconds(time: datetime.time):
    hours = time.hour * 60 * 60
    minutes = time.minute * 60
    seconds = time.second
    return (hours+ minutes + seconds)

