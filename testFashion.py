import csv
import pandas as pd

# d = open("J:\\Python computer vision\\CNN-CV\\data_fashion\\styles.csv", 'r')
# data = csv.reader(d)
# ans = []


# data = pd.read_csv("J:\\Python computer vision\\CNN-CV\\data_fashion\\styles.csv")
# print(len(data))

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
    #print(lijst['articleType'].unique())
    # apparel: topwear, bottomwear, dress,




    #for i in columns:
    #    print(f"unique {i}:")
    #    t = lijst[i].unique()
    #    print(t)