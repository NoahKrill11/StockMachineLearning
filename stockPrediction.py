
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd

start_date = '2010-01-01'
end_date = '2021-1-26'

df = data.DataReader('AAPL', 'yahoo', start_date, end_date)
df = df[['Close']] 

forecast_out = 30 #'n=30' days
#Create another column (the target ) shifted 'n' units up
df['Prediction'] = df[['Close']].shift(-forecast_out)
# create independent data set x and convert to numpy
X = np.array(df.drop(['Prediction'],1))
#Remove the last '30' rows
X = X[:-forecast_out]

#Create dependent data set and convert to numoy array
Y=np.array(df['Prediction'])
Y=Y[:-forecast_out]

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=.2)

lr = LinearRegression()
lr.fit(x_train,y_train)

#testing the model 
lr_confidence = lr.score(x_test,y_test)
print("lr confidence = ", lr_confidence)

#set forcast = tp last 30 rows of the original adj close
x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
#print (x_forcast)

lr_prediction = lr.predict(x_forecast )
#svm_prediction = svr_rbf.predict(x_forecast )

print(lr_prediction)
#print(svm_prediction)