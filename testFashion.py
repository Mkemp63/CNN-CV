import csv
import pandas as pd
import cv2

# d = open("J:\\Python computer vision\\CNN-CV\\data_fashion\\styles.csv", 'r')
# data = csv.reader(d)
# ans = []


# data = pd.read_csv("J:\\Python computer vision\\CNN-CV\\data_fashion\\styles.csv")
# print(len(data))

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
print(batch_data[0])
print(batch_data[1])
columns = batch_data[0]
data = batch_data[1:]
lijst = pd.DataFrame(data, columns=columns)

masterCategories = lijst['masterCategory'].unique()
print(masterCategories)
for mc in masterCategories:
    t = lijst.loc[lijst['masterCategory'] == mc]
    print(f"{mc} uniques: ")
    t2 = t['subCategory'].unique()
    print(t2)
    for sc in t2:
        tt = t.loc[t['subCategory'] == sc]
        print(f"{sc} uniques: {tt['articleType'].unique()}")


t = lijst.loc[lijst['masterCategory'] == masterCategories[7]]
print(t.size)
# print(lijst['articleType'].unique())
# apparel: topwear, bottomwear, dress,

# 'id', 'gender', 'masterCategory', 'subCategory', 'articleType']
data0 = lijst.loc[lijst['articleType'].isin(['Tops','Tshirts'])]
data1 = lijst.loc[lijst['articleType'] == 'Trousers']
data2 = lijst.loc[lijst['articleType'] == 'Sweaters']
data3 = lijst.loc[lijst['articleType'] == 'Dresses']
data4 = lijst.loc[lijst['articleType'].isin(['Rain Jacket', 'Jackets'])]
data5 = lijst.loc[lijst['articleType'] == 'Sandals']
data6 = lijst.loc[lijst['articleType'] == 'Shirts']
data7 = lijst.loc[lijst['articleType'] == 'Sports Shoes']
data8 = lijst.loc[lijst['subCategory'] == 'Bags']
data9 = lijst.loc[lijst['articleType'] == 'Heels']
print("Sizes .....")
print(data0.shape[0])
print(f"uniques: {data0['articleType'].unique()}")
print(data4)
print(data1.shape[0])
print(data2.shape[0])
print(data3.shape[0])
print(data4.shape[0])
print(data5.shape[0])
print(data6.shape[0])
print(data7.shape[0])
print(data8.shape[0])
print(data9.shape[0])
print("TEST")


def getImage(dataSet):
    lijst = []
    for index, r in dataSet.iterrows():
        filename = r.iloc[0] + ".jpg"
        img = cv2.imread(f"J:\\Python computer vision\\CNN-CV\\data_fashion\\converted/" + filename, 0)
        lijst.append(img)
    return lijst

ans1 = getImage(data0)
print(len(ans1))

# for filename in os.listdir("J:\\Python computer vision\\CNN-CV\\data_fashion\\images"):


    #for i in columns:
    #    print(f"unique {i}:")
    #    t = lijst[i].unique()
    #    print(t)