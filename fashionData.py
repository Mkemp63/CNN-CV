import csv
import pandas as pd
import cv2
import numpy as np

def getImage(dataSet):
    lijst = []
    for index, r in dataSet.iterrows():
        filename = r.iloc[0] + ".jpg"
        img = cv2.imread(f"J:\\Python computer vision\\CNN-CV\\data_fashion\\converted/" + filename, 0)
        if img is None:
            print(f"None! {filename}")
        else:
            img = img.reshape((28, 28, 1))
            lijst.append(img)

    lijst = np.array(lijst)
    return lijst


def getFashionData():
    batch_data = []
    with open("J:\\Python computer vision\\CNN-CV\\data_fashion\\styles.csv", newline='', encoding='utf8') as f:
        csvread = csv.reader(f)
        batch_data = list(csvread)

    for row in batch_data:
        lengte = len(row)
        for i in range(0, len(row)):
            index = lengte - 1 - i
            if (index > 4):
                del row[index]

    columns = batch_data[0]
    data = batch_data[1:]
    lijst = pd.DataFrame(data, columns=columns)

    data0 = lijst.loc[lijst['articleType'].isin(['Tops', 'Tshirts'])]
    data1 = lijst.loc[lijst['articleType'] == 'Trousers']
    data2 = lijst.loc[lijst['articleType'] == 'Sweaters']
    data3 = lijst.loc[lijst['articleType'] == 'Dresses']
    data4 = lijst.loc[lijst['articleType'].isin(['Rain Jacket', 'Jackets'])]
    data5 = lijst.loc[lijst['articleType'] == 'Sandals']
    data6 = lijst.loc[lijst['articleType'] == 'Shirts']
    data7 = lijst.loc[lijst['articleType'] == 'Sports Shoes']
    data8 = lijst.loc[lijst['subCategory'] == 'Bags']
    data9 = lijst.loc[lijst['articleType'] == 'Heels']

    images0 = getImage(data0)
    images1 = getImage(data1)
    images2 = getImage(data2)
    images3 = getImage(data3)
    images4 = getImage(data4)
    images5 = getImage(data5)
    images6 = getImage(data6)
    images7 = getImage(data7)
    images8 = getImage(data8)
    images9 = getImage(data9)

    return images0, images1, images2, images3, images4, images5, images6, images7, images8, images9

