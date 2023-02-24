import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load training data
x = -50 + np.random.random((25000,1))*100
y = x**2

print("X:")
print(len(x))
print("Y:")
print(len(y))
#exit()

# Define model
model = Sequential()
model.add(Dense(40, input_dim=1, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x, y, epochs=50, batch_size=50)

predictions = model.predict([10, 5, 200, 13])
print(predictions) # Approximately 100, 25, 40000, 169