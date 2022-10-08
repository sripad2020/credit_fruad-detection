import pandas as pd
data=pd.read_csv('creditcard.csv')
print(data.columns)
print()
print(data.describe())
print()
print(data.isna().sum())
print()
print(data.corr())
import seaborn as sn
import matplotlib.pyplot as plt
sn.heatmap(data.corr())
plt.show()
sn.countplot(data['class'])
plt.show()
q1=data['V1'].quantile(0.25)
q3=data['V1'].quantile(0.75)
q_1=data['V2'].quantile(0.25)
q_3=data['V2'].quantile(0.75)
qu_1=data['V3'].quantile(0.25)
qu_3=data['V3'].quantile(0.75)
qua_1=data['V4'].quantile(0.25)
qua_3=data['V4'].quantile(0.75)
quant_1=data['V5'].quantile(0.25)
quant_3=data['V5'].quantile(0.75)
quanti_1=data['V6'].quantile(0.25)
quanti_3=data['V6'].quantile(0.75)

#a=data[(data['V1'] < q3) |  (data['V1'] >q1)|(data['V2'] > q_3)|(data['V2']>q_1) | (data['V3'] <qu_3) | (data['V3'] > qu_1) | (data['V4'] < qua_3)|(data['V4'] >qua_1) | (data['V5'] < quant_3)|(data['V5'] >quant_1)|(data['V6'] < quanti_3)|(data['V6'] > quanti_1)].values
from sklearn.preprocessing import MinMaxScaler
stand=MinMaxScaler()
x=data.iloc[:,1:29].values
y=data['class'].values
x=stand.fit_transform(x)
y=pd.DataFrame(y)
print(y)
y=stand.fit_transform(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.25)
import keras.activations,keras.losses
from keras.models import  Sequential
from keras.layers import Dense
model=Sequential()
model.add(Dense(units=x_train.shape[1],input_dim=x.shape[1],activation=keras.activations.sigmoid))
model.add(Dense(units=x_train.shape[1],activation=keras.activations.sigmoid))
model.add(Dense(units=x_train.shape[1],activation=keras.activations.sigmoid))
model.add(Dense(units=x_train.shape[1],activation=keras.activations.sigmoid))
model.add(Dense(units=1,activation=keras.activations.sigmoid))
model.compile(optimizer='adam',loss=keras.losses.binary_crossentropy,metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=20,epochs=25)
pred=model.predict(x_test)
print(pred)
evalueation=model.evaluate(x_test,y_test)
print(evalueation)

import sweetviz as sv
sv_rp=sv.analyze(data)
sv_rp.show_html('report.html')

from autoviz.AutoViz_Class import AutoViz_Class
av=AutoViz_Class()
df=av.AutoViz('creditcard.csv')