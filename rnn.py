# Recurrent Neural Networks

# Part 1 - Data Preprocessing

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv(r'C:\Users\firoj\Downloads\P16-Recurrent-Neural-Networks\Recurrent_Neural_Networks\Google_Stock_Price_Train.csv')
# Create a set of data for model to learn from 
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
# Normalize independent variables to values betweeen 0 and 1
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    # Get a set of 60 stock prices
    X_train.append(training_set_scaled[i-60:i, 0])
    # Get stock price of day following the 60th day
    y_train.append(training_set_scaled[i, 0])
# Convert X_train and y_train into numpy arrays
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping X_train for it to be accepted as input for RNN
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Part 2 - Building the RNN

# Importing Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initializing the RNN
regressor = Sequential()

# Adding the first Long Short Term Memory (LSTM) layer 
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
# Dropout regularization for potential overfitting (20% of neurons in LSTM layer will be ignored during training) 
regressor.add(Dropout(0.2))

# Adding a second LSTM layer 
regressor.add(LSTM(units = 50, return_sequences = True))
# Dropout regularization for potential overfitting
regressor.add(Dropout(0.2))

# Adding a third LSTM layer 
regressor.add(LSTM(units = 50, return_sequences = True))
# Dropout regularization for potential overfitting
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer
regressor.add(LSTM(units = 50))
# Dropout regularization for potential overfitting
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


# Part 3 - Making predictions and visualising results

# Getting the real stock prices of 2017
dataset_test = pd.read_csv(r'C:\Users\firoj\Downloads\P16-Recurrent-Neural-Networks\Recurrent_Neural_Networks\Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
# Concatenate 'open' prices in training set and test set
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
# Get stock price of first financial day of 2017 minus 60 to last stock price in the concatenated dataset
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
# Reshape and scale input for it to be accepted by RNN
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])   
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# RNN makes prediction
predicted_stock_price = regressor.predict(X_test)
# Obtain original values of prediction 
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
# Graph real stock prices vs. predicted stock prices for January 2017
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time (Financial Days)')
plt.ylabel('Google Stock Price ($)')
plt.legend()
plt.show()
