'''
数据预处理

1. 'OverallQual', 'GrLivArea'这两个变量与'SalePrice'有很强的线性关系;
2. 'GarageCars', 'GarageArea'与'SalePrice'也有很强的线性关系，但'GarageCars', 'GarageArea'相关性0.88，有很强的共线性，只取其一即可，取与目标变量关系更强的'GarageCars'；
3. 同样地，'TotalBsmtSF', '1stFlrSF'也有很强的共线性，只取其一即可，取'TotalBsmtSF'；
4. 因此，选取的特征有：'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt'。

'''
import pandas as pd
from sklearn.model_selection import train_test_split

class Data_Processing():
    def __init__(self, data):
        self.data = data

    def data_processing(self):
        # isnull() boolean, isnull().sum()统计所有缺失值的个数
        # isnull().count()统计所有项个数（包括缺失值和非缺失值），.count()统计所有非缺失值个数
        total = self.data.isnull().sum().sort_values(ascending=False)
        percent = (self.data.isnull().sum() / self.data.isnull().count()).sort_values(ascending=False)
        # pd.concat() axis=0 index,axis=1 column, keys列名
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        # print(missing_data.head(20))

        # 处理缺失值，将含缺失值的整列剔除
        data1 = self.data.drop(missing_data[missing_data['Total'] > 1].index, axis=1)
        # data1.to_csv('data1.csv')
        median = data1['SalePrice'].median()
        # print('房价中位数：', median)
        data1.loc[data1['SalePrice'] >= median, 'SalePrice'] = median + 1  # 1 广义的代表高房价
        # print(data1)
        data1.loc[data1['SalePrice'] < median, 'SalePrice'] = median - 1  # median-1 广义的代表低房价
        # print(data1)

        # 由于特征Electrical只有一个缺失值，故只需删除该行即可
        data2 = data1.drop(data1.loc[data1['Electrical'].isnull()].index)
        # 检查缺失值数量
        # print('缺失值数量：', data2.isnull().sum().max())

        feature_data = data2.drop(['SalePrice'], axis=1)
        target_data = data2['SalePrice'].astype(str)  # 将分类特征转化为字符型

        # print(feature_data.shape)
        # print(target_data.shape)

        # 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt'。
        sub_feature_data = {
            'OverallQual': feature_data['OverallQual'],  # 分类特征
            'GrLivArea': feature_data['GrLivArea'],
            'GarageCars': feature_data['GarageCars'],
            'TotalBsmtSF': feature_data['TotalBsmtSF'],
            'FullBath': feature_data['FullBath'],
            'TotRmsAbvGrd': feature_data['TotRmsAbvGrd'],
            'YearBuilt': feature_data['YearBuilt']
        }
        sub_feature_data = pd.DataFrame(sub_feature_data)
        sub_feature_data['OverallQual'] = sub_feature_data['OverallQual'].astype(str)  # 将分类特征转化为字符型

        # print(sub_feature_data)
        # print(target_data)

        # print(sub_feature_data.shape)
        # print(target_data)
        # 将数据集划分为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(sub_feature_data, target_data, test_size=0.3)

        return X_train, X_test, y_train, y_test

if __name__ == '__main__':

    data = pd.read_csv("data.csv")
    dp = Data_Processing(data)

    X_train, X_test, y_train, y_test = dp.data_processing()

    # print(X_train)
