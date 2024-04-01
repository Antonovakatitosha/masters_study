# Import OpenCV library
import cv2
import numpy as np

# Open/Read input image
img = cv2.imread("strong shapes.jpeg")

# Splitting the image into different color channels
b, g, r = cv2.split(img)
cv2.imshow("Red", r)
cv2.imshow("Green", g)
cv2.imshow("Blue", b)
# Displays the output window untill any key is pressed
cv2.waitKey(0)

zeros = np.zeros(img.shape[:2], dtype="uint8")
cv2.imshow("Red", cv2.merge([zeros, zeros, r]))
cv2.imshow("Green", cv2.merge([zeros, g, zeros]))
cv2.imshow("Blue", cv2.merge([b, zeros, zeros]))
cv2.waitKey(0)

cv2.destroyAllWindows()
