import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf 
from tensorflow.keras.layers import LSTM,Dense, Input
from tensorflow.keras.models import Model

def split_data_into_frames(data,samples_to_predict,samples_to_train):
    '''
        samples_to_predict != samples_to_train
    '''
    s = data.shape[0]
    data_ = []
    data_y = []
    for i in range(samples_to_train,s):
        data_.append(data[i-samples_to_train:i,:])
        data_y.append(data[i:i+samples_to_predict,:])
    return np.array(data_), np.array(data_y)

def train_test_split(data, precentage=0.8):
    s = data.shape[0]
    train_len = int(precentage*s)
    return data[:train_len, :], data[train_len:,:]
 
path = r"E:\Experments\AI_projects\Dataset\portfolio_data.csv"
data = pd.read_csv(path)

print(data.describe())
print(data.head())

print(data.info)
data.drop(['Date'], axis=1, inplace=True)
for index,i in enumerate(data.columns):
    plt.subplot(4,1,index+1)
    data[i].plot()
    plt.ylabel("Price")
plt.show()

for i in data.columns:
    print(f"Total Earn of {i}:", data[i].sum(), "$")

mean_data = pd.DataFrame([])
days = 50
for i in data.columns:
    mean_data[i] = data[i].rolling(days).mean()

mean_data.plot(subplots=True,title="50 days mean")
data.plot(subplots=True, title="orignal")
plt.show()

plt.title("Stocks stability")
rate_change_each_day_origina = data.pct_change()
rate_change_each_day_origina.plot(subplots=True, title="Stocks stability")
plt.show()

data.hist()
plt.show()


sns.jointplot(x='AMZN', y='DPZ', data=data, kind='scatter', color='seagreen')
plt.show()


sns.pairplot(data, kind='reg')
plt.show()

sns.heatmap(data.corr(), annot=True)
plt.show()

# get all data into numpy 
data = data.values

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

print(scaled_data.shape)

train, test = train_test_split(scaled_data)
print(train.shape, test.shape)

samples_to_be_predicted = 1
train_x,train_y = split_data_into_frames(train,samples_to_be_predicted,60)
test_x,test_y = split_data_into_frames(test,samples_to_be_predicted,60)



n_feature = train_x.shape[-1]
t_steps = train_x.shape[1]

train_y = train_y.reshape(train_y.shape[0],train_y.shape[-1])
test_y = test_y.reshape(test_y.shape[0],test_y.shape[-1])

print(f"train_x: {train_x.shape}, train_y: {train_y.shape}, test_x: {test_x.shape}, test_y: {test_y.shape}")

inp = Input(shape=(t_steps,n_feature))
x = LSTM(128, return_sequences=True)(inp)
x = LSTM(64)(x)
x = Dense(32)(x)
output = Dense(4)(x)

model = Model(inputs=inp,outputs=output)
model.compile(loss="mean_squared_error", optimizer="adam")

print(model.summary())

Epochs = 100
batch_size = 32

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
model.fit(train_x,train_y,
        validation_data=(test_x,test_y),
        batch_size=batch_size,
        epochs=Epochs,
        callbacks=[callback],
        verbose=1)

path_models = r"E:\Experments\AI_projects\models"

model.save(path_models+"\\Lstm_pridect.h5")

predicted_data = model.predict(test_x)
rmse = np.sqrt(np.mean(((predicted_data - test_y) ** 2)))


predicted_data = scaler.inverse_transform(predicted_data)
print("this is rmse: ", rmse)

train = scaler.inverse_transform(train)

total_data = np.append(train[:,1],predicted_data[:,1])
plt.plot(total_data)
plt.plot(train[:,1])
plt.show()