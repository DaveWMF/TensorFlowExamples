import tensorflow as tf
import cv2 as cv
import numpy as np

model = tf.keras.models.load_model("ModeloNumeros.h5")

nueve = cv.imread('./nueve.png', cv.IMREAD_GRAYSCALE)
nueveArray = cv.resize(nueve, (28,28))
nueveArray = np.invert(np.array([nueveArray]))


print("\n\nPredicción:")
prediccion = model.predict([nueveArray])
print('Creo que es '+str(np.argmax(prediccion)))