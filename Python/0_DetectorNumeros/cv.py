import cv2 as cv

nueve = cv.imread('./nueve.png', cv.IMREAD_GRAYSCALE)
print(nueve)
cv.imshow("Cosa", nueve)
nueveArray = cv.resize(nueve, (28,28))
cv.waitKey(0)