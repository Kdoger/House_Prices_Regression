'''
数据预处理

1. 'OverallQual', 'GrLivArea'这两个变量与'SalePrice'有很强的线性关系;
2. 'GarageCars', 'GarageArea'与'SalePrice'也有很强的线性关系，但'GarageCars', 'GarageArea'相关性0.88，有很强的共线性，只取其一即可，取与目标变量关系更强的'GarageCars'；
3. 同样地，'TotalBsmtSF', '1stFlrSF'也有很强的共线性，只取其一即可，取'TotalBsmtSF'；
4. 因此，选取的特征有：'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt'。

'''
import pandas as pd
from sklearn.model_selection import train_test_split

def init_data():
    data = pd.read_csv("data.csv")
    median = data['SalePrice'].median()
    classLabels = []
    for i in range(data.shape[0]):
        if data['SalePrice'][i] >= median:   # 高房价
            classLabels.append('高房价')
        else:
            classLabels.append('低房价')
    dataMatIn = {
        'GrLivArea': data['GrLivArea'],
        'YearBuilt': data['YearBuilt']
    }
    # dataMatIn = np.insert(dataMatIn, 0, 1, axis=1)  #特征数据集，添加1是构造常数项x0
    return dataMatIn, classLabels

if __name__ == '__main__':
    data = pd.read_csv("data.csv")

    # isnull() boolean, isnull().sum()统计所有缺失值的个数
    # isnull().count()统计所有项个数（包括缺失值和非缺失值），.count()统计所有非缺失值个数
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)
    # pd.concat() axis=0 index,axis=1 column, keys列名
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data.head(20))

    # 处理缺失值，将含缺失值的整列剔除
    data1 = data.drop(missing_data[missing_data['Total'] > 1].index, axis=1)
    # 由于特征Electrical只有一个缺失值，故只需删除该行即可
    data2 = data1.drop(data1.loc[data1['Electrical'].isnull()].index)
    # 检查缺失值数量
    print('缺失值数量：', data2.isnull().sum().max())

    feature_data = data2.drop(['SalePrice'], axis=1)
    target_data = data2['SalePrice']
    #'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt'。
    sub_feature_data = {
        'OverallQual': feature_data['OverallQual'],
        'GrLivArea': feature_data['GrLivArea'],
        'GarageCars': feature_data['GarageCars'],
        'TotalBsmtSF': feature_data['TotalBsmtSF'],
        'FullBath': feature_data['FullBath'],
        'TotRmsAbvGrd': feature_data['TotRmsAbvGrd'],
        'YearBuilt': feature_data['YearBuilt']
    }
    sub_feature_data = pd.DataFrame(sub_feature_data
                                    )
    print(sub_feature_data)
    print(target_data)

    # 将数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(sub_feature_data, target_data, test_size=0.3)

    df_train = pd.concat([X_train, y_train], axis=1)
    print(df_train)
