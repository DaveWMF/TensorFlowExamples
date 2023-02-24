import tensorflow as tf
import numpy as np
import random as rand

print('TensoFlow version: {}'.format(tf.__version__))


datos = -20 + np.random.random((25000, 1))*40
data_train = np.array([datos, datos*10 ])
data_test = np.array([np.array([0.6,0.7,0.8,0.9,1, 1.1, 1.2]),np.array([6, 7, 8, 9, 10, 11, 12])])

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])

model.fit(data_train[0], data_train[1], epochs=25, batch_size=50)

model.evaluate(data_test[0], data_test[1], verbose=2)

print("\n\nPredicción:")
prediccion = model.predict([2, 4, 6, 8, 10])
print('Creo que es '+str(prediccion))