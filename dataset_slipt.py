'''
数据集：1460行，81列

'''
import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv("data.csv")
    print(data)