import cv2
import os

def adaptFashionImages():
    aantal = 0
    for filename in os.listdir("J:\\Python computer vision\\CNN-CV\\data_fashion\\images"):
        aantal += 1
        img = cv2.imread(f"J:\\Python computer vision\\CNN-CV\\data_fashion\\images/"+filename, 0)
        img = cv2.resize(img, (28, 28))
        cv2.imwrite(f"J:\\Python computer vision\\CNN-CV\\data_fashion\\converted/{filename}", img)
        if aantal % 1000 == 0:
            print(aantal)
