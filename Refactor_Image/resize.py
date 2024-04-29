import os
import cv2 as cv

path = "C:\\Users\\untiv\\Desktop\\data"
for file in os.listdir(path):
    file_path = os.path.join(path, file)
    # print(file_path)
    image = cv.imread(file_path)
    resized_image = cv.resize(image, (300, 360))
    cv.imwrite(file_path, resized_image)
