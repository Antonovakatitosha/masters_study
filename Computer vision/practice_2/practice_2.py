import cv2
import numpy as np

# Загрузка изображения и преобразование в градации серого
img = cv2.imread('shapes.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Применение пороговой обработки для получения бинарного изображения
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# Поиск контуров
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Нанесение контуров на исходное изображение
cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

# Отображение изображения
cv2.imshow('Contours', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
