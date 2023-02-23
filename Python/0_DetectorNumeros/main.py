import tensorflow as tf
import cv2 as cv
import numpy as np

print('TensoFlow version: {}'.format(tf.__version__))
#print('TensoFlow version: {}'.format(ts.version))

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)


nueve = cv.imread('./nueve.png', cv.IMREAD_GRAYSCALE)
nueveArray = cv.resize(nueve, (28,28))
#nueveArray = nueveArray.reshape(-1, 28,28,1)
nueveArray = np.invert(np.array([nueveArray]))


print("\n\nPredicción:")
prediccion = model.predict([nueveArray])
print('Creo que es '+str(np.argmax(prediccion)))